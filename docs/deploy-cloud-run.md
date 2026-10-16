# Deploying to Google Cloud Run

The API runs on Cloud Run at https://churn-api-995576020018.us-central1.run.app (`/docs` for Swagger). Every push to `main` deploys it through `.github/workflows/deploy.yml`, with no stored service-account key.

| Setting | Value |
|---|---|
| Project | `finsight-gpu` |
| Region | `us-central1` |
| Service | `churn-api` |
| Resources | 1 vCPU, 1 GiB, min 0 / max 3 instances, port 8000 |
| Deployer identity | `github-churn-deployer` service account, `roles/run.sourceDeveloper` |
| Auth | Workload Identity Federation (GitHub OIDC), trusted for this repo's `main` only |

## What the workflow does

1. **test**: `python -m pytest src/serving` (41 tests) against `requirements-serve.txt`, then `python -m evals.run --api-url local` (economics, parity, contract, latency).
2. **frontend**: `npm ci`, typecheck, lint, vitest, build in `frontend/` (the UI itself deploys on Vercel).
3. **build**: builds the `Dockerfile` and pushes it to `ghcr.io/<owner>/churn-api` on pushes to `main` (build only on PRs).
4. **deploy** (pushes to `main` only): authenticates with WIF, then runs `deploy-cloudrun` with `source: ./`. Cloud Build builds the same `Dockerfile` into Artifact Registry, because Cloud Run cannot pull from `ghcr.io`. A `curl /health` smoke test follows, then `python -m evals.run` against the new revision.

The deploy job is skipped when the `GCP_WIF_PROVIDER` / `GCP_DEPLOY_SA` repository variables are unset, e.g. on forks.

## One-time setup

```bash
PROJECT=finsight-gpu
REPO=<github-owner>/ai-customer-retention-mlops
SA=github-churn-deployer@$PROJECT.iam.gserviceaccount.com

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com iamcredentials.googleapis.com --project $PROJECT

gcloud iam service-accounts create github-churn-deployer --project $PROJECT
gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:$SA" --role="roles/run.sourceDeveloper"

# Workload Identity pool + GitHub OIDC provider, restricted to this repo's main branch
gcloud iam workload-identity-pools create github --location=global --project $PROJECT
gcloud iam workload-identity-pools providers create-oidc github-repo \
  --location=global --workload-identity-pool=github --project $PROJECT \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --attribute-condition="assertion.repository=='$REPO' && assertion.ref=='refs/heads/main'"

# Let tokens from that repo impersonate the deployer
gcloud iam service-accounts add-iam-policy-binding $SA --project $PROJECT \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/<project-number>/locations/global/workloadIdentityPools/github/attribute.repository/$REPO"
```

Pool and provider names above are examples; use whatever your project already has. Then, in GitHub → Settings → Secrets and variables → Actions → **Variables**:

| Variable | Value |
|---|---|
| `GCP_WIF_PROVIDER` | `projects/<project-number>/locations/global/workloadIdentityPools/<pool>/providers/<provider>` |
| `GCP_DEPLOY_SA` | `github-churn-deployer@finsight-gpu.iam.gserviceaccount.com` |

**Public access:** the workflow does not pass `--allow-unauthenticated`. The public `roles/run.invoker` binding for `allUsers` was set once on the service and persists across revisions, so the deployer never needs IAM-admin rights.

## API key (optional)

The public demo runs without one. For a private deployment, keep the key in Secret Manager and expose it as `API_KEY`; the POST endpoints then require an `X-API-Key` header, and `/health` stays open for probes:

```bash
printf '%s' "$(openssl rand -hex 32)" | gcloud secrets create churn-api-key --data-file=- --project finsight-gpu
gcloud run services update churn-api --region us-central1 --project finsight-gpu --update-secrets=API_KEY=churn-api-key:latest
```

The service's runtime service account needs `roles/secretmanager.secretAccessor` on the secret. The browser UI can't hold a key, so with one set the UI only works through a server-side proxy. The post-deploy eval reads `API_KEY` from the environment, so add it as a GitHub secret. Set `CORS_ALLOW_ORIGINS` to the UI's origin at the same time.

## Cold starts

`--min-instances=0` keeps the service free when idle, so the first request after a quiet period takes a few seconds. The React UI waits up to 12 s on `/health`, then falls back to a labelled Demo Mode. Before a live demo, either hit `/health` a minute early or temporarily run:

```bash
gcloud run services update churn-api --region us-central1 --project finsight-gpu --min-instances=1
```

and set it back to 0 afterwards (an always-on instance is billed).

## Streamlit

The Streamlit Community Cloud app (`src/serving/app_streamlit_cloud.py`) loads `artifacts/champion/model.pkl` directly and does not call this API.
