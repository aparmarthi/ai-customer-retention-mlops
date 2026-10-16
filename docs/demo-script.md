# Demo Script: Churn Decision Intelligence (3–5 minutes)

**Story:** score a customer → why they're flagged → what to do → what it's worth at our scale → hand the exec a one-page summary.

**Setup (before the call):** open https://churn.ameyparmarthi.com/ about a minute early so the Cloud Run API wakes from scale-to-zero. The status pill should be green; if it's amber, the app is in Demo Mode and every estimate is badged. Have a second tab on `#/simulator`.

---

## 1. Frame the problem (30 s): `#/overview`

> "In the holdout window, about 1.2% of subscribers churn, and this UI treats each scoring cycle as a month. Retention can't call everyone, so the question isn't 'who churns?' but 'who do we contact this cycle, and is it worth it?' This is the tool an operator and an exec would use for that."

Point at the champion (LightGBM, ROC AUC 0.966, PR AUC 0.539 on a time-based holdout of subscribers whose last transaction is after 2017-02-24, about 43× the base rate) and say every dollar figure is simulated on that validation window, not realised revenue.

## 2. Score a customer (45 s): `#/score` → Single

Pick a high-risk profile, such as "One-off purchase, expired, inactive", and score it. Check the band in a rehearsal first: the live model's score can differ from the profile's rule-of-thumb estimate.

> "This comes from the live model on Cloud Run. It lands in the **Contact now** band, so the suggested action is personal outreach with a retention incentive."

## 3. Why they're flagged (45 s): the "Why this score" panel

> "Operators won't act on a number they can't explain. These are the model's own attributions: TreeSHAP contributions from the LightGBM champion, returned by the API and converted to probability points. Red raises risk, green lowers it."

Say what they are and aren't:

> "They explain the score, not what causes churn. They cost about 50 ms per call, so `/predict` includes them by default for interactive use and batch scoring skips them. If the API is asleep, the app falls back to Demo Mode, and the panel switches to indicative drivers from a transparent rule of thumb and says so in its heading."

Optional: paste `{}` into the JSON editor and score it. It comes back at 0.99 with a warning that most features are missing: "the model reads missing activity as churn, so a renamed column upstream would flood the contact list. The API flags it instead of hiding it."

## 4. What to do at volume (30 s): Batch

Click **Use 5 samples** or paste the 60-customer JSON. Show the rank, the band and the "Top reasons" chips (indicative: batch scoring skips attributions), then **Export CSV**.

> "The export is a contact list: rank, band, suggested action and top reasons per customer, ready to load into an outreach tool."

## 5. What it's worth at our scale (60 s): `#/simulator`

Set subscribers to the prospect's base (the default is 1,000,000). Every figure on the tab is at that scale. Start at K = 1,478, the ROI-optimal 0.68 threshold, then move the value per save and the save rate.

> "At $120 per save, $5 per contact and a 20% save rate, contacting the top 0.76% of a 1M base (about 7,650 people) nets about $91K a cycle, roughly $1.1M a year. That's a 239% net ROI: every $1 of contact spend brings back $3.39. The line under the panel is the measured number it scales from: $17,666 on the 193,205-subscriber validation window. Save rate and value per save tie in the sensitivity chart because net value scales with their product. We can look up what a subscriber is worth, but the save rate is pure assumption, so that's what we test first."

Scroll to **Which policy should we run?**:

> "Here's the trap. The retention team's capacity is the top 5.2% (10,000 on the validation window). Give all of them the $5 incentive and you lose about $35K a cycle. Capacity only pays with a cheap channel: the incentive for the top 0.76% plus a $0.50 nudge for the rest of the top 5.2% nets about $126K. That's why policy v2 is tiered: the API returns `contact_now`, `nudge` or `monitor` with every score, and the 0.68 threshold stays the default label."

## 6. Hand the exec a one-pager (45 s): **Executive summary**

Click **Executive summary**.

> "This is what goes to the VP: the recommendation, the key numbers, a contact plan by risk band at their subscriber count, assumptions, risks and caveats. Print to PDF or download as Markdown. The link at the bottom reopens this exact scenario."

Close on the risk line:

> "The 20% save rate is an assumption. The next step is a holdout experiment to measure it before we scale."

---

## Prepared Q&A

**"Are these real model explanations?"**

> "With the live API, yes: exact TreeSHAP contributions from the LightGBM model via `pred_contrib=True`, so no new dependency. All 38 contributions plus the base value sum to the score's log-odds, and a test checks that. The panel shows the top 5. In Demo Mode it switches to a transparent rule of thumb and labels it 'indicative drivers', because I won't present an approximation as the model."

**"What did explanations cost?"**

> "Latency. p50 goes from 3 ms to about 51 ms in-process, because TreeSHAP walks 146 trees with up to 1,212 leaves each. That's fine for an agent looking at one subscriber, so it's on by default for `/predict`, with `explain: false` to skip it. A 50,000-row batch would pay that 50,000 times, so batch scoring never explains."

**"Is 0.45 a 45% chance of churning?"**

> "No. The scores rank well but aren't calibrated: the nudge tier, scores 0.21 to 0.68, actually churned at 8.9% on validation. The model trained at 6.4% prevalence and was evaluated at 1.24%. The economics use observed precision per tier, never raw scores, so the dollar figures hold. Calibration, e.g. isotonic on a later window, is the next step, and the eval harness tracks this as an info check."

**"How would you know if it breaks in production?"**

> "Labels arrive about 30 days late, so I watch what doesn't need labels. Every batch of 1,000 or more returns a PSI of its tier mix against validation: stable, moderate or significant shift. Mostly-missing records are counted and flagged. After every deploy, CI re-runs the eval harness against Cloud Run: score parity on five reference records, the input contract and latency."

**"How would you secure it for a customer?"**

> "Set `API_KEY` from Secret Manager and the POST endpoints require it. The caller is the CRM or a batch job, never a browser. For stronger isolation, use Cloud Run IAM auth or internal ingress, plus CORS restricted to the UI. Deploys are already keyless through Workload Identity Federation, and the logs carry scores and counts, never feature values. The public demo runs without a key on purpose, since a key in a browser is public."

**"Why not leave-one-out for the drivers?"**

> "It saturates. For a customer at 99.7% risk, removing a strong factor moves the score about 1 point, so every driver looks negligible and they don't add up. Scaling each factor's log-odds by the sigmoid's average slope fixes both problems."

**"Is the $91K (or $17,666) real?"**

> "It's a simulation, not realised revenue. $17,666 is computed on the 193,205-subscriber time-based validation window from the model's actual ranking under stated assumptions. $91K is the same thing scaled linearly to a 1M base, which assumes the same churn rate and risk mix. The holdout experiment is what turns it into a measured number."

**"Why does the 3.4× differ from the 239%?"**

> "3.4× is gross: retained value divided by contact spend. Net ROI subtracts the spend first: $17,666 ÷ $7,390 = 2.39, or 239%. I show both and label them so nobody double-counts."

**"How do the bands map to the policy?"**

> "Contact now is score ≥ 0.68, the ROI-optimal threshold (top 1,478 at validation scale). Nudge runs down to 0.2058, the score at the retention team's 10,000-contact capacity, through a $0.50 channel. Everyone else is monitored and re-scored next cycle. Those tiers live in a versioned artifact, `deployment_policy_v2.json`, which the API loads and returns as `action`."
