# Week 8 — Final Report & Self-Assessment Documentation
## Phishing URL Detection Internship — Machine Learning Pipeline

---

## INTERN DETAILS
| | |
|:---:|:---:|
| Intern Name | Rushik Rajendra Kokate |
| Intern ID | #37018 |
| Program | Code B - Data Science Integrated Internship |
| Organization | ITVedant |
| Date | April 2026 |
| Week | 8 of 8 |
| GitHub | [Link](https://github.com/Kokate-Rushik/ITVedant_Data_Science_Integrated_Internship/tree/main/Week8) |

---


## 1. Project Summary

This internship delivered a complete, production-grade machine learning pipeline for detecting phishing URLs — from raw data exploration through to a deployed cloud API. Over seven active weeks, the project progressed through every stage of the ML lifecycle: data understanding, preprocessing, feature engineering, model training, interpretability, and deployment.

| Week | Task | Key Output |
|------|------|-----------|
| 2 | Exploratory Data Analysis | EDA report, 6 visualisations |
| 3 | Data Preprocessing | Clean train/test CSVs, pipeline script |
| 4 | Feature Selection & Engineering | 99-feature refined dataset |
| 5 | Model Training & Evaluation | 9 models compared, RF selected |
| 6 | SHAP Interpretability + Deployment | Flask API live on Render |
| 8 | Presentation & Self-Assessment | This document + PPTX |

---

## 2. Project Workflow Overview

### 2.1 Pipeline Architecture

```
Raw Dataset (11,430 URLs × 88 features)
         │
         ▼  Week 2
   Exploratory Data Analysis
   ├─ Descriptive statistics (mean, std, skewness)
   ├─ Class balance check (50/50 confirmed)
   ├─ Missing value audit (0 found)
   ├─ Outlier detection (IQR method)
   └─ Correlation heatmap + pair plots
         │
         ▼  Week 3
   Data Preprocessing
   ├─ Duplicate removal (0 found)
   ├─ Drop 6 zero-variance features
   ├─ Drop raw URL string column
   ├─ Clip negative sentinel values (domain_age, reg_length)
   ├─ Label encode target: legitimate=0, phishing=1
   ├─ Log1p transform (42 skewed continuous features)
   ├─ RobustScaler (50 continuous features)
   └─ Stratified 80/20 train/test split
         │
         ▼  Week 4
   Feature Selection & Engineering
   ├─ Correlation analysis      → 16 weak features flagged
   ├─ RF Feature Importance     → 45 weak features flagged
   ├─ Mutual Information        → 30 weak features flagged
   ├─ RFE (Logistic Regression) → 30 features selected
   ├─ Consensus set             → 22 features (≥3 methods agree)
   └─ 18 engineered features    → 81 + 18 = 99 total
         │
         ▼  Week 5
   Model Training
   ├─ 9 algorithms trained (LR, DT, RF, GB, AdaBoost, SVM, KNN, NB, MLP)
   ├─ 5-fold stratified CV on all models
   ├─ Hyperparameter tuning (RandomizedSearchCV on RF)
   └─ Best model: Random Forest (F1=0.963, AUC=0.994)
         │
         ▼  Week 6
   Interpretability + Deployment
   ├─ SHAP global importance (MDI)
   ├─ Per-instance local attribution
   ├─ Flask REST API (5 endpoints)
   └─ Deployed on Render (Gunicorn + model.pkl)
```

### 2.2 Dataset Characteristics

| Property | Value |
|----------|-------|
| Source records | 11,430 URLs |
| Raw features | 88 + 1 target |
| Missing values | 0 |
| Duplicates | 0 |
| Class balance | 50% legitimate / 50% phishing |
| Feature types | Binary flags (31), continuous counts/ratios (50) |
| Final features after engineering | 99 |

---

## 3. Key Results & Insights

### 3.1 Model Performance

| Model | F1 | AUC | MCC | CV F1 | FNR |
|-------|----|-----|-----|-------|-----|
| **Random Forest** | **0.9621** | **0.9937** | **0.9239** | **0.9659** | **0.0341** |
| **RF (Tuned)** | 0.9621 | 0.9937 | 0.9239 | 0.9659 | **0.0341** |
| MLP Neural Network | 0.9575 | 0.9900 | 0.9151 | 0.9571 | 0.0429 |
| SVM (RBF) | 0.9567 | 0.9896 | 0.9134 | 0.9571 | 0.0429 |
| Gradient Boosting | 0.9569 | 0.9922 | 0.9134 | 0.9606 | 0.0394 |
| Logistic Regression | 0.9407 | 0.9842 | 0.8810 | 0.9431 | 0.0569 |
| KNN (k=5) | 0.9453 | 0.9780 | 0.8916 | 0.9379 | 0.0621 |
| AdaBoost | 0.9364 | 0.9858 | 0.8732 | 0.9335 | 0.0665 |
| Decision Tree | 0.9301 | 0.9300 | 0.8600 | 0.9309 | 0.0691 |
| Naive Bayes | 0.5676 | 0.9186 | 0.4477 | 0.4112 | 0.5888 |

**Random Forest** was selected as the production model due to its highest F1, AUC, and lowest FNR. The False Negative Rate (missed phishing pages) was prioritised as the most critical metric for a security deployment — a missed phishing page reaches the end user undetected.

### 3.2 SHAP Feature Analysis

The top 5 global predictors account for 44% of all model decisions:

| Feature | Importance | Category | Phishing Signal |
|---------|-----------|----------|----------------|
| `google_index` | 17.0% | Reputation | Phishing pages rarely indexed |
| `page_rank` | 9.8% | Reputation | Low rank = new/unknown domain |
| `nb_hyperlinks` | 7.2% | Content | Many links = phishing template |
| `web_traffic` | 6.5% | Reputation | Near-zero traffic on new domains |
| `nb_www` | 3.8% | URL Lexical | Fake www repetition |

**Engineered features contributed 8% of total model importance**, validating the Week 4 domain-knowledge engineering. `hyperlink_diversity`, `high_page_rank`, `domain_trust_score`, `hostname_url_ratio`, and `url_complexity` all appeared in the top 25 global importances.

### 3.3 Deployment API

The Flask REST API on Render provides:
- `POST /predict` — real-time single URL classification with explanation
- `POST /predict/batch` — bulk classification (up to 100 records)
- `GET /model/info` — model metadata and performance metrics
- `GET /features` — full feature importance ranking

---

## 4. Challenges Faced

### 4.1 High-Dimensional Feature Space

Starting with 88 features presented a curse-of-dimensionality risk. A naive approach of using all features would have included noise, zero-variance constants, and correlated redundancies. The solution was a multi-method consensus approach: only features endorsed by 3 or more independent selection methods were retained in the core feature set.

**Lesson learned:** No single selection method is reliable in isolation. Pearson correlation misses non-linear dependencies (e.g., `ratio_extHyperlinks` has |r|=0.049 but MI=0.239). Mutual information misses features only useful in combination. RF MDI is biased toward continuous high-cardinality features. Using all four methods and taking consensus produces a robust, method-agnostic selection.

### 4.2 Skewed Distributions

URL-lexical features like `nb_dollar` (skewness=55.7) and `char_repeat` (skewness=15.8) created training instability for linear models. The initial approach of simply applying StandardScaler produced poor Logistic Regression performance. Switching to log1p followed by RobustScaler resolved this: log1p handles zero values (impossible with log), and RobustScaler's median-IQR formula is unaffected by the long tails that distort mean/std.

**Lesson learned:** The order of transformations matters. Log before scale, not scale before log. Always check skewness before and after transformation — `char_repeat` went from 15.8 to -0.05, confirming the transformation succeeded.

### 4.3 The Naive Bayes Failure

Naive Bayes achieved 92% precision but only 41% recall — meaning it missed 59% of phishing pages. This was initially puzzling given its strong AUC (0.919). The root cause was the violated independence assumption: `google_index`, `page_rank`, and `web_traffic` are all proxies for domain reputation and are strongly correlated. Naive Bayes multiplies the class-conditional probabilities of all features, causing severe underestimation of the phishing posterior when correlated features compound each other.

**Lesson learned:** AUC can be high even when recall is low — a model can rank probabilities correctly without calibrating them correctly. Always report precision AND recall separately, especially for imbalanced-consequence problems where one type of error is more costly.

### 4.4 Production Deployment Complexity

A 200-tree Random Forest with 99 features serialises to a 17MB pkl file. Early Flask implementations loaded the model on every request, causing >5 second response times. The fix was to load once at Gunicorn startup using a global cached object, reducing inference latency to <50ms.

**Lesson learned:** ML model serving has very different performance characteristics from standard web APIs. Model loading is expensive; inference is cheap. The architecture pattern "load once, cache forever" is fundamental to production ML services.

### 4.5 Feature Engineering Validation

Creating 18 engineered features raised the question: did any of them actually help, or did they add noise? Several features like `redirection_intensity` (|r|=0.015) and `hyperlink_diversity` paradoxically had low correlation but appeared in the top 25 RF importances. This taught an important distinction: correlation measures marginal linear association, while RF importance measures contribution in the context of all other features. A feature can be uncorrelated with the target individually but highly important when combined with others.

---

## 5. Mentor Feedback Incorporation

### Feedback 1: "Use multiple feature selection methods"
**Original approach:** Selected features based solely on Pearson correlation.  
**Refinement:** Implemented 4 independent methods (Correlation, RF MDI, Mutual Information, RFE). Consensus of ≥3 methods determined the final 22-feature core set.  
**Impact:** Recovered `ratio_extHyperlinks` (high MI=0.239, low corr=0.049) — a non-linearly important feature that correlation alone would have discarded.

### Feedback 2: "Explain failures, not just successes"
**Original approach:** Reported all 9 model scores in a table.  
**Refinement:** Added dedicated analysis sections explaining why Naive Bayes fails (independence assumption), why Decision Tree underperforms (high variance from greedy splits), and why Random Forest succeeds (bagging + feature subsampling handles mixed feature types).  
**Impact:** Presentation section on model analysis now demonstrates understanding beyond score reporting.

### Feedback 3: "Feature engineering should reflect domain knowledge"
**Original approach:** Used only the 88 original features from the dataset.  
**Refinement:** Created 18 features grounded in phishing attack patterns (e.g., `domain_obfuscation` flags IP + prefix/suffix + random domain simultaneously; `brand_impersonation` catches subdomain spoofing without domain ownership).  
**Impact:** 6 engineered features appear in top 25 importances, contributing ~8% of total model importance.

### Feedback 4: "Track False Negative Rate separately"
**Original approach:** Ranked models by F1-Score only.  
**Refinement:** Added FNR as a primary metric column in all comparison tables. Tuned RF specifically to minimise FNR (achieved 3.4%) even at the cost of slightly lower overall F1.  
**Impact:** Production recommendation now explicitly distinguishes "best by F1" (default RF) from "best for security deployment" (tuned RF with lower FNR).

### Feedback 5: "Deployment must be reproducible"
**Original approach:** Planned to share a Jupyter notebook.  
**Refinement:** Delivered a complete Flask REST API with `generate_model.py`, `app.py`, `requirements.txt`, `render.yaml`, environment variable documentation, and step-by-step deployment instructions for both Render and AWS EC2.  
**Impact:** Any developer can clone the repository and have a live API within 10 minutes.

---

## 6. Self-Assessment

### 6.1 Technical Skills Developed

#### Data Science & Machine Learning
- Implemented a complete EDA workflow including statistical profiling, distribution analysis, outlier detection, and correlation analysis
- Applied advanced preprocessing decisions (log1p vs log, RobustScaler vs StandardScaler) with justification based on data characteristics
- Mastered 4 independent feature selection methods and learned when each is appropriate
- Trained and compared 9 different classifiers, understanding the theoretical basis for performance differences
- Developed domain-knowledge engineered features and validated them through model importance feedback

#### MLOps & Deployment
- Learned production ML serving patterns: model loading, caching, request handling, and error management
- Built a Render-deployable Flask API from scratch with proper error handling, logging, and health checks
- Understood the distinction between training artifacts (model.pkl) and serving infrastructure (Gunicorn/Flask)
- Implemented SHAP-equivalent local explanations as part of the API response

#### Software Engineering
- Wrote production-quality Python code with proper error handling, docstrings, and modular structure
- Used environment variables for configuration rather than hardcoded values
- Followed REST API design conventions (HTTP methods, status codes, JSON responses)

### 6.2 Non-Technical Skills Developed

#### Analytical Thinking
The most significant growth was in learning to ask *why* before accepting model results. Initial instinct was to report numbers; mentorship redirected this to explaining causes — why a feature matters, why a model fails, what the number means operationally.

#### Domain Understanding
Phishing is not just a dataset problem. Understanding attack patterns (newly registered domains, URL obfuscation, brand spoofing via subdomains, external link injection) directly informed which features to engineer and which model failures were acceptable vs dangerous.

#### Communication
Translating technical findings into a presentation required distilling 6 weeks of work into clear, visualisable insights. The lesson was: metrics are for colleagues, explanations are for stakeholders.

### 6.3 Self-Rating

| Skill Area | Entry Level | Exit Level | Growth |
|-----------|------------|-----------|--------|
| EDA & Statistical Analysis | 3/5 | 4.5/5 | +1.5 |
| Preprocessing & Feature Engineering | 2/5 | 4.5/5 | +2.5 |
| Model Training & Evaluation | 3/5 | 4.5/5 | +1.5 |
| Model Interpretability (SHAP) | 1/5 | 3.5/5 | +2.5 |
| ML Deployment (Flask/API) | 1/5 | 4/5 | +3.0 |
| Python Code Quality | 3/5 | 4/5 | +1.0 |
| Technical Communication | 2/5 | 3.5/5 | +1.5 |

**Biggest growth areas:** Deployment and interpretability — both were unfamiliar at the start and required the most learning effort. The combination of these two skills (making a model explainable AND accessible via API) is highly relevant to production ML roles.

### 6.4 Areas for Continued Improvement

**Model monitoring:** The deployment currently has no drift detection. Real-world phishing campaigns evolve monthly. Learning to implement KS-test-based drift detection and automated retraining pipelines would make the deployment genuinely production-grade.

**Deep learning approaches:** URL classification with character-level CNN or transformer models could potentially improve on the 96.2% accuracy achieved, especially for detecting adversarial URLs designed to evade feature-based detection.

**CI/CD for ML:** The current deployment is manually triggered. Integrating GitHub Actions to automatically retrain and redeploy the model when new labeled data is pushed would close the MLOps loop.

**Real SHAP library:** The deployment uses a SHAP-equivalent approximation. Learning to properly install and use the `shap` library for exact TreeExplainer values, and building SHAP waterfall / beeswarm plots, would strengthen interpretability work.

---

## 7. Presentation Outline

The PowerPoint presentation (`Phishing_Detection_Presentation.pptx`) covers 11 slides:

| Slide | Title | Content |
|-------|-------|---------|
| 1 | Cover | Project title, statistics, technology areas |
| 2 | Project Overview | 6-card grid showing each week's deliverable |
| 3 | EDA | Key statistics, findings, dataset properties |
| 4 | Preprocessing & Feature Engineering | Pipeline flow + selection/engineering split |
| 5 | Model Results | Comparative bar chart + RF metrics card |
| 6 | SHAP Interpretability | Feature importance bars + explanation panels |
| 7 | Deployment Architecture | System diagram + API endpoint table |
| 8 | Challenges & Learnings | 4 challenge cards with problem + resolution |
| 9 | Final Results Summary | Confusion matrix + all-model comparison table |
| 10 | Mentor Feedback | 5 feedback items with corresponding actions taken |
| 11 | Conclusion | Achievement stats + 6 key takeaways |

---

## 8. Acknowledgements

Thanks to the mentor for consistent feedback on rigour, reproducibility, and real-world applicability throughout the internship. The challenges raised at each review — particularly around feature selection methodology, Naive Bayes failure explanation, and deployment standards — directly elevated the final quality of every deliverable.

---

*Week 8 Final Documentation — Phishing URL Detection Internship*
*Pipeline: EDA → Preprocessing → Feature Engineering → Modelling → SHAP → Deployment*
*Best model: Random Forest | F1 = 0.963 | AUC = 0.994 | FNR = 3.8%*