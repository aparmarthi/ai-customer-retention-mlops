# Deploying to Google Cloud Run

One-time setup. After this, every `git push main` auto-deploys via GitHub Actions.

## Prerequisites
- Google Cloud account (free tier: 2M requests/month, 360,000 GB-seconds compute)
- `gcloud` CLI installed

---

## 1. Create a GCP project and enable APIs

```bash
gcloud projects create churn-mlops --name="Churn MLOps"
gcloud config set project churn-mlops

gcloud services enable \
  run.googleapis.com \
  containerregistry.googleapis.com \
  artifactregistry.googleapis.com
```

## 2. Create a service account for GitHub Actions

```bash
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions deployer"

# Grant permission to deploy to Cloud Run
gcloud projects add-iam-policy-binding churn-mlops \
  --member="serviceAccount:github-actions@churn-mlops.iam.gserviceaccount.com" \
  --role="roles/run.admin"

# Grant permission to pull images from GitHub Container Registry
gcloud projects add-iam-policy-binding churn-mlops \
  --member="serviceAccount:github-actions@churn-mlops.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# Export the key
gcloud iam service-accounts keys create gcp-key.json \
  --iam-account=github-actions@churn-mlops.iam.gserviceaccount.com
```

## 3. Add the key to GitHub Secrets

In your GitHub repo → Settings → Secrets and variables → Actions → New secret:

| Secret name | Value |
|---|---|
| `GCP_SA_KEY` | Paste the entire contents of `gcp-key.json` |

**Delete `gcp-key.json` locally after adding it** — never commit it.

## 4. Push to main

```bash
git add .
git commit -m "add: Dockerfile + GitHub Actions CI/CD deploy to Cloud Run"
git push origin main
```

GitHub Actions will:
1. Run tests (`pytest src/serving/test_policy.py`)
2. Build the Docker image and push to `ghcr.io`
3. Deploy to Cloud Run

Your API will be live at:
`https://churn-api-<hash>-uc.a.run.app`

## 5. Point Streamlit Cloud at the Cloud Run URL

In Streamlit Community Cloud → App settings → Secrets:
```toml
CHURN_API_URL = "https://churn-api-<hash>-uc.a.run.app"
```

---

## Why Cloud Run over Render free tier

| | Render free | Cloud Run free tier |
|---|---|---|
| Sleep after idle | Yes, 15 min | No (scales to 0 but wakes in ~2s) |
| Free requests | 750 hrs/month | 2 million/month |
| Docker support | Yes | Yes (required) |
| Custom domain | Paid | Free (CNAME) |
| Interview credibility | Moderate | High — it's what prod teams use |
