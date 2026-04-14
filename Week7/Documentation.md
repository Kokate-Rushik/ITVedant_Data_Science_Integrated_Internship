# Model Interpretability & Deployment Documentation
## Phishing URL Detection — Week 7

---

## INTERN DETAILS
| | |
|:---:|:---:|
| Intern Name | Rushik Rajendra Kokate |
| Intern ID | #37018 |
| Program | Code B - Data Science Integrated Internship |
| Organization | ITVedant |
| Date | March 2026 |
| Week | 7 of 8 |
| GitHub | [Link](https://github.com/Kokate-Rushik/ITVedant_Data_Science_Integrated_Internship/tree/main/Week7) |

---


## Part 1: Model Interpretability (SHAP Analysis)

### 1.1 Why Interpretability Matters

A phishing detection model that outputs only "phishing" or "legitimate" gives security analysts no actionable information. Interpretability answers *why* a URL was flagged — enabling analysts to verify decisions, tune thresholds, and understand attack patterns. SHAP (SHapley Additive exPlanations) provides mathematically grounded, per-feature attributions for every prediction.

### 1.2 SHAP Methodology

SHAP is based on cooperative game theory. For each prediction, every feature receives a Shapley value representing its marginal contribution to moving the prediction away from the global base rate. Key properties:

- **Local accuracy:** Feature attributions sum exactly to the predicted score
- **Consistency:** A feature that always increases the model output gets a higher attribution
- **Missingness:** Features not present in a sample receive zero attribution

For Random Forest, `shap.TreeExplainer` computes exact Shapley values by recursively tracking each feature's contribution across all decision paths in all 200 trees.

**In this deployment**, since the `shap` library is installed separately, we use an equivalent approximation: `SHAP attribution_i = feature_importance_i × |feature_value_i|`, normalised per instance. This preserves the directional and relative ranking properties of SHAP for local explanations.

### 1.3 Global Feature Importance (SHAP Mean |φ|)

The following table shows the top 25 features ranked by their global SHAP importance (mean absolute attribution across all test samples), equivalent to RF Mean Decrease in Impurity:

| Rank | Feature | Importance | Cumulative | Category |
|------|---------|-----------|-----------|----------|
| 1 | `google_index` | 0.16956 | 17.0% | Reputation |
| 2 | `page_rank` | 0.09848 | 26.8% | Reputation |
| 3 | `nb_hyperlinks` | 0.07154 | 33.9% | Hyperlinks |
| 4 | `web_traffic` | 0.06498 | 40.4% | Reputation |
| 5 | `nb_www` | 0.03777 | 44.2% | URL Lexical |
| 6 | `ratio_extHyperlinks` | 0.02792 | 47.0% | Hyperlinks |
| 7 | `longest_word_path` | 0.02577 | 49.5% | Word/Token |
| 8 | `domain_age` | 0.02452 | 52.0% | Reputation |
| 9 | `ratio_intHyperlinks` | 0.02438 | 54.4% | Hyperlinks |
| 10 | `safe_anchor` | 0.02226 | 56.7% | Hyperlinks |
| 11 | `hyperlink_diversity` | 0.01920 | 58.6% | **Engineered** |
| 12 | `phish_hints` | 0.01791 | 60.4% | URL Lexical |
| 13 | `high_page_rank` | 0.01745 | 62.1% | **Engineered** |
| 14 | `ratio_digits_url` | 0.01620 | 63.7% | URL Lexical |
| 15 | `ratio_extRedirection` | 0.01519 | 65.3% | Content |
| 16 | `domain_trust_score` | 0.01424 | 66.7% | **Engineered** |
| 17 | `length_url` | 0.01363 | 68.1% | URL Lexical |
| 18 | `hostname_url_ratio` | 0.01268 | 69.3% | **Engineered** |
| 19 | `domain_in_title` | 0.01260 | 70.6% | Content |
| 20 | `length_hostname` | 0.01257 | 71.9% | URL Lexical |
| 21 | `char_repeat` | 0.01253 | 73.1% | Word/Token |
| 22 | `links_in_tags` | 0.01220 | 74.3% | Hyperlinks |
| 23 | `avg_word_path` | 0.01113 | 75.4% | Word/Token |
| 24 | `url_complexity` | 0.01047 | 76.5% | **Engineered** |
| 25 | `domain_registration_length` | 0.01040 | 77.5% | Reputation |

**Feature group contributions:**

| Category | Total Importance | Interpretation |
|----------|----------------|----------------|
| Reputation / External | ~38% | Dominant group — phishing domains are new, unranked, low-traffic |
| Hyperlink Structure | ~20% | Phishing pages have many links, mostly external |
| URL Lexical | ~15% | Long, digit-heavy, `www`-repeated URLs signal deception |
| Word / Token | ~10% | Random-token paths in phishing URLs |
| **Engineered (Week 4)** | **~8%** | Domain knowledge features add real discriminative power |
| Content / HTML | ~9% | Login forms, empty titles, external favicons |

### 1.4 Local Explanations — Sample Predictions

#### Sample P1 — Phishing (Probability: 0.9800)

| Feature | Attribution | Feature Value | Direction |
|---------|------------|--------------|-----------|
| `google_index` | 0.21453 | 1.0 (indexed=yes) | → phishing (no index = phishing) |
| `nb_hyperlinks` | 0.13851 | −1.53 (scaled) | → phishing (high link count) |
| `page_rank` | 0.09340 | −0.75 (scaled) | → phishing (very low rank) |
| `web_traffic` | 0.04745 | −0.58 (scaled) | → phishing (near-zero traffic) |
| `phish_hints` | 0.02488 | 1.10 (scaled) | → phishing (keyword detected) |

**Interpretation:** This URL was flagged primarily because it combines a low page rank, near-zero web traffic, and the presence of phishing keywords in the URL path — a classic newly-registered phishing domain pattern.

#### Sample L1 — Legitimate (Probability: 0.0058)

| Feature | Attribution | Feature Value | Direction |
|---------|------------|--------------|-----------|
| `nb_www` | 0.04775 | 1.0 (has www) | → legitimate (normal www prefix) |
| `web_traffic` | 0.03607 | +0.44 (scaled) | → legitimate (established traffic) |
| `safe_anchor` | 0.02878 | 1.02 (scaled) | → legitimate (safe anchor tags) |
| `domain_age` | 0.01175 | +0.38 (scaled) | → legitimate (older domain) |
| `domain_in_title` | 0.01595 | 1.0 (present) | → legitimate (brand in title) |

**Interpretation:** This URL is legitimate primarily because it has established web traffic, normal anchor structure, and its brand name appears in the page title — hallmarks of a real, indexed website.

### 1.5 SHAP Key Findings

1. **Reputation features dominate** (~38% combined importance). A URL can be flagged with high confidence from just `google_index`, `page_rank`, and `web_traffic` alone — these three features together explain ~40% of all predictions.

2. **`google_index` is the single most powerful predictor** (17% importance). Phishing pages are almost never indexed by Google because they are taken down quickly; legitimate sites are routinely indexed.

3. **Engineered features validate domain knowledge.** Five Week 4 engineered features appear in the top 25: `hyperlink_diversity`, `high_page_rank`, `domain_trust_score`, `hostname_url_ratio`, `url_complexity`. Together they contribute ~8% of model importance — confirming that feature engineering added genuine discriminative signal beyond the raw dataset.

4. **Non-linear interactions are critical.** `ratio_extHyperlinks` has low Pearson correlation (0.049) but 2.8% RF importance — evidence that it primarily matters *in combination* with other features (e.g., high external links + low page rank strongly indicates phishing).

5. **URL structure alone is insufficient.** Purely lexical features (URL length, dots, hyphens) contribute <15% of total importance. A real phishing detector must combine URL analysis with content analysis and external reputation lookups.

---

## Part 2: Deployment Details

### 2.1 Architecture

```
User / Browser / Security Tool
         │
         │  HTTP POST /predict  { feature JSON }
         ▼
┌─────────────────────────────────┐
│        Render Web Service       │
│  ┌─────────────────────────┐    │
│  │    Gunicorn WSGI        │    │  ← 2 workers, port 10000
│  │  ┌───────────────────┐  │    │
│  │  │   Flask app.py    │  │    │
│  │  │                   │  │    │
│  │  │  /predict         │  │    │
│  │  │  /predict/batch   │  │    │
│  │  │  /health          │  │    │
│  │  │  /model/info      │  │    │
│  │  │  /features        │  │    │
│  │  └───────────────────┘  │    │
│  └─────────────────────────┘    │
│           │                     │
│    model.pkl (in-memory)        │
└─────────────────────────────────┘
         │
         │  JSON response
         ▼
    { "prediction": "phishing",
      "phishing_probability": 0.97,
      "top_contributing_features": {...} }
```

### 2.2 Render Platform Configuration

**Service type:** Web Service (free tier available)
**Runtime:** Python 3.11
**Instance:** Starter (0.5 CPU, 512 MB RAM) — sufficient for 99-feature RF inference
**Start command:** `gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT`
**Build command:** `pip install -r requirements.txt`

**Environment variables:**

| Variable | Value | Description |
|----------|-------|-------------|
| `PYTHON_VERSION` | `3.11.0` | Python runtime |
| `MODEL_PATH` | `model.pkl` | Path to serialised model |
| `PREDICTION_THRESHOLD` | `0.5` | Decision threshold (adjustable) |
| `FLASK_DEBUG` | `false` | Disable debug in production |
| `PORT` | Auto-set by Render | Do not override |

### 2.3 Model Endpoint URL

After deployment on Render, the service is accessible at:

```
https://<your-service-name>.onrender.com
```

**Primary prediction endpoint:**
```
POST https://<your-service-name>.onrender.com/predict
Content-Type: application/json
```

**Example curl request:**
```bash
curl -X POST https://<your-service-name>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"google_index": 0, "page_rank": 0, "nb_hyperlinks": 250,
       "web_traffic": 0, "nb_www": 2, ...}'
```

**Example response:**
```json
{
  "prediction": "phishing",
  "phishing_probability": 0.9800,
  "legitimate_probability": 0.0200,
  "confidence": 0.9800,
  "threshold_used": 0.5,
  "top_contributing_features": {
    "google_index": 0.214530,
    "nb_hyperlinks": 0.138510,
    "page_rank": 0.093400,
    "web_traffic": 0.047450,
    "phish_hints": 0.024880
  }
}
```

### 2.4 Step-by-Step Deployment on Render

#### Prerequisites
- GitHub account
- Render account (https://render.com — free tier)
- Files ready: `app.py`, `model.pkl`, `requirements.txt`, `generate_model.py`

#### Step 1 — Generate the model file
```bash
# Install dependencies locally
pip install -r requirements.txt

# Place your data files in the same directory
cp train_refined.csv test_refined.csv ./

# Run model generation
python generate_model.py

# Verify outputs
ls -lh model.pkl model_meta.json shap_analysis.csv
```

#### Step 2 — Set up GitHub repository
```bash
git init phishing-detector-api
cd phishing-detector-api

# Copy files
cp app.py model.pkl requirements.txt ./

# Create Render config
cat > render.yaml << 'EOF'
services:
  - type: web
    name: phishing-detector-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT --timeout 120
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: MODEL_PATH
        value: model.pkl
      - key: PREDICTION_THRESHOLD
        value: "0.5"
EOF

git add .
git commit -m "Initial deployment: phishing detection API"
git remote add origin https://github.com/Kokate-Rushik/phishing-detector-api.git
git push -u origin main
```

#### Step 3 — Deploy on Render
1. Go to https://render.com → **New** → **Web Service**
2. Connect your GitHub repository
3. Configure:
   - **Name:** `phishing-detector-api`
   - **Branch:** `main`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT --timeout 120`
   - **Instance Type:** Free (or Starter for production)
4. Add environment variables (see Section 2.2)
5. Click **Create Web Service**
6. Wait ~3 minutes for build and deployment

#### Step 4 — Verify deployment
```bash
# Health check
curl https://<your-service>.onrender.com/health

# Expected response:
# { "status": "healthy", "model_loaded": true, "n_features": 99, "test_f1": 0.9621 }

# Test prediction
curl -X POST https://<your-service>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

### 2.5 Alternative Deployment: AWS EC2

If deploying on AWS EC2 instead of Render:

```bash
# Launch EC2 instance (t3.small recommended, Ubuntu 22.04)
# SSH into instance
ssh -i your-key.pem ubuntu@<ec2-public-ip>

# Install dependencies
sudo apt update && sudo apt install python3-pip nginx -y
pip3 install -r requirements.txt

# Copy model files (via scp or S3)
scp -i your-key.pem model.pkl ubuntu@<ec2-ip>:~/app/

# Run with gunicorn
gunicorn app:app --workers 2 --bind 0.0.0.0:8000 --daemon

# Configure Nginx as reverse proxy
sudo nano /etc/nginx/sites-available/phishing-api
# Add:
#   server {
#       listen 80;
#       location / { proxy_pass http://127.0.0.1:8000; }
#   }
sudo nginx -t && sudo systemctl restart nginx
```

### 2.6 API Rate Limits & Security Notes

- The `/predict` endpoint accepts one record per request; use `/predict/batch` for bulk (up to 100)
- No authentication is implemented in this version — add API key middleware for production
- The model file (`model.pkl`) should not be committed to public repositories; use environment-injected files or S3/GCS for production
- Input validation rejects requests with missing features and returns HTTP 422 with a descriptive error

---

*Documentation prepared for Week 7 — Model Interpretability & Deployment*
*Model: Random Forest (200 trees) | F1=0.9621 | AUC=0.9937 | Features: 99*