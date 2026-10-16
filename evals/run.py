"""
Evaluation harness for the churn decision service.

Offline checks (always run, no data or network needed) recompute the business figures and
the tier calibration from the committed champion artifacts. Live checks call the API:
score parity on reference records, input contract and guardrails, and latency.

Usage:
  python -m evals.run                                   # offline only (CI gate)
  python -m evals.run --api-url local                   # + in-process API (current code)
  python -m evals.run --api-url https://churn-api-995576020018.us-central1.run.app

Writes evals/results/<UTC timestamp>_<target>.json and exits 1 if any check fails.
Set API_KEY in the environment if the target API requires it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "champion"
EVALS = ROOT / "evals"
logger = logging.getLogger("evals")

Check = Dict[str, Any]


def _check(name: str, passed: bool, **details: Any) -> Check:
    return {"check": name, "passed": bool(passed), **details}


def _load(name: str) -> Dict[str, Any]:
    return json.loads((ART / name).read_text())


# ── offline ──────────────────────────────────────────────────────────────────
def economics_checks() -> List[Check]:
    """Recompute net value per policy from tier counts: net = TP x S x V - K x C."""
    pol = _load("deployment_policy_v2.json")
    v, summary = pol["value_per_save"], pol["validation_summary"]
    tiers = {t["action"]: t for t in pol["tiers"]}
    counts, churners = summary["tier_counts"], summary["tier_churners"]

    def net(name: str, cost: float, save: float) -> float:
        return churners[name] * save * v - counts[name] * cost

    t1 = net("contact_now", tiers["contact_now"]["cost_per_contact"], tiers["contact_now"]["assumed_save_rate"])
    t2 = net("nudge", tiers["nudge"]["cost_per_contact"], tiers["nudge"]["assumed_save_rate"])
    t2_full = net("nudge", tiers["contact_now"]["cost_per_contact"], tiers["contact_now"]["assumed_save_rate"])
    reported = summary["net_per_cycle"]
    return [
        _check("net_threshold_only", abs(t1 - reported["threshold_only"]) < 1, computed=round(t1), reported=reported["threshold_only"]),
        _check("net_top_10k_single_channel", abs(t1 + t2_full - reported["top_10k_single_channel"]) < 1,
               computed=round(t1 + t2_full), reported=reported["top_10k_single_channel"]),
        _check("net_tiered", abs(t1 + t2 - reported["tiered"]) < 1, computed=round(t1 + t2), reported=reported["tiered"]),
    ]


def lift_checks() -> List[Check]:
    """Lift = precision at K (or PR-AUC) over the holdout churn rate."""
    m = _load("metrics.json")
    base = m["valid_churn_rate"]
    documented = {"pr_auc": 43.4, "5000": 21.6, "10000": 14.5}
    computed = {"pr_auc": m["pr_auc"] / base, **{k: m["precision_at_k"][k] / base for k in ("5000", "10000")}}
    return [
        _check(f"lift_{k}", abs(round(computed[k], 1) - documented[k]) < 0.05, computed=round(computed[k], 2), documented=documented[k])
        for k in documented
    ]


def calibration_checks() -> List[Check]:
    """Observed churn rate per tier vs the tier's score range.

    Only tier-level counts are committed, so this is a bound check, not a reliability curve:
    a calibrated model's observed rate must fall inside the score range of the tier.
    """
    pol = _load("deployment_policy_v2.json")
    counts = pol["validation_summary"]["tier_counts"]
    churners = pol["validation_summary"]["tier_churners"]
    bounds = [t["min_score"] for t in pol["tiers"]]
    checks = []
    for i, tier in enumerate(pol["tiers"]):
        name = tier["action"]
        lo, hi = bounds[i], (1.0 if i == 0 else bounds[i - 1])
        observed = churners[name] / counts[name]
        checks.append(_check(
            f"calibration_{name}", lo <= observed < hi or (i == 0 and observed >= lo),
            score_range=[lo, hi], observed_churn_rate=round(observed, 4), severity="info",
        ))
    return checks


# ── live ─────────────────────────────────────────────────────────────────────
def make_client(api_url: str) -> Callable[..., Any]:
    """Return call(method, path, **kwargs) -> (status, json) for a URL or the in-process app.

    Sends X-API-Key when the API_KEY env var is set, matching the API's optional key check.
    """
    headers = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}
    if api_url == "local":
        from fastapi.testclient import TestClient
        from src.serving.api import app
        client = TestClient(app)
        client.__enter__()  # runs startup (loads artifacts)
        send = client.request
    else:
        import requests
        session = requests.Session()
        send = lambda m, p, **kw: session.request(m, api_url.rstrip("/") + p, timeout=60, **kw)  # noqa: E731

    def call(method: str, path: str, **kwargs: Any) -> tuple[int, Any]:
        r = send(method, path, headers=headers, **kwargs)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, None

    return call


def parity_checks(call: Callable[..., Any]) -> List[Check]:
    """The deployed model must reproduce the scores recorded for the reference records."""
    refs = json.loads((EVALS / "reference_records.json").read_text())
    checks = []
    for ref in refs:
        status, body = call("POST", "/predict", json={"record": ref["record"]})
        got = body.get("churn_probability") if status == 200 else None
        checks.append(_check(f"parity: {ref['label']}", got is not None and abs(got - ref["expected"]) <= 0.001,
                             expected=ref["expected"], got=None if got is None else round(got, 4)))
    return checks


def contract_checks(call: Callable[..., Any]) -> List[Check]:
    """Input contract and guardrails: bad input must fail loudly, never silently."""
    rec = json.loads((EVALS / "reference_records.json").read_text())[0]["record"]
    s_empty, b_empty = call("POST", "/predict", json={"record": {}})
    s_rec, b_rec = call("POST", "/predict", json={"record": rec})
    cases = [
        ("health_ok", call("GET", "/health")[0] == 200),
        ("explanations_returned", s_rec == 200 and len(b_rec.get("explanations", [])) == 5),
        ("action_returned", s_rec == 200 and b_rec.get("action") in {"contact_now", "nudge", "monitor"}),
        ("empty_record_warned", s_empty == 200 and bool(b_empty.get("warnings"))),
        ("threshold_out_of_range_422", call("POST", "/predict", json={"record": rec, "threshold": 1.5})[0] == 422),
        ("top_k_on_single_400", call("POST", "/predict", json={"record": rec, "policy": "top_k"})[0] == 400),
        ("batch_bad_body_422", call("POST", "/predict_batch", json={"not_records": []})[0] == 422),
        ("batch_empty_400", call("POST", "/predict_batch", json={"records": []})[0] == 400),
        ("top_k_without_k_400", call("POST", "/predict_batch", json={"records": [rec], "policy": "top_k"})[0] == 400),
    ]
    return [_check(f"contract: {name}", ok) for name, ok in cases]


def latency_checks(call: Callable[..., Any], n: int, p95_budget_ms: float) -> List[Check]:
    """Sequential /predict latency (client-measured), with and without explanations, plus one 1,000-record batch."""
    rec = json.loads((EVALS / "reference_records.json").read_text())[0]["record"]
    call("GET", "/health")  # absorb a cold start before timing

    def timed(body: Dict[str, Any]) -> List[float]:
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            call("POST", "/predict", json=body)
            out.append((time.perf_counter() - t0) * 1000)
        return out

    with_expl, without = timed({"record": rec}), timed({"record": rec, "explain": False})
    t0 = time.perf_counter()
    status, _ = call("POST", "/predict_batch", json={"records": [rec] * 1000})
    batch_ms = (time.perf_counter() - t0) * 1000
    p50, p95 = (float(x) for x in np.percentile(with_expl, [50, 95]))
    return [
        _check("latency_predict_p95", p95 <= p95_budget_ms, n=n, p50_ms=round(p50, 1), p95_ms=round(p95, 1),
               max_ms=round(max(with_expl), 1), budget_ms=p95_budget_ms),
        _check("latency_predict_no_explain_p50", True, p50_ms=round(float(np.percentile(without, 50)), 1), severity="info"),
        _check("latency_batch_1000", status == 200, total_ms=round(batch_ms, 1), per_record_ms=round(batch_ms / 1000, 2)),
    ]


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-url", help="'local' for the in-process app, or a base URL")
    parser.add_argument("--n", type=int, default=50, help="Sequential /predict calls for latency")
    parser.add_argument("--p95-budget-ms", type=float, default=500.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")  # keep API request logs out of the report
    logger.setLevel(logging.INFO)

    checks = economics_checks() + lift_checks() + calibration_checks()
    api_version = None
    if args.api_url:
        call = make_client(args.api_url)
        api_version = call("GET", "/")[1].get("version")
        checks += parity_checks(call) + contract_checks(call) + latency_checks(call, args.n, args.p95_budget_ms)

    failed = [c for c in checks if not c["passed"] and c.get("severity") != "info"]
    target = "offline" if not args.api_url else "local" if args.api_url == "local" else "remote"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {"run_at_utc": stamp, "target": args.api_url or "offline", "api_version": api_version,
              "passed": not failed, "n_checks": len(checks), "n_failed": len(failed), "checks": checks}
    out = EVALS / "results" / f"{stamp}_{target}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    for c in checks:
        mark = "PASS" if c["passed"] else ("INFO" if c.get("severity") == "info" else "FAIL")
        extra = {k: v for k, v in c.items() if k not in {"check", "passed", "severity"}}
        logger.info("%s  %-48s %s", mark, c["check"], extra or "")
    logger.info("\n%d checks, %d failed -> %s", len(checks), len(failed), out.relative_to(ROOT))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
