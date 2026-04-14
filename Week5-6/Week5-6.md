# Model Training & Evaluation Report
## Phishing URL Detection Dataset — Week 5 Assignment

---

## INTERN DETAILS
| | |
|:---:|:---:|
| Intern Name | Rushik Rajendra Kokate |
| Intern ID | #37018 |
| Program | Code B - Data Science Integrated Internship |
| Organization | ITVedant |
| Date | March 2026 |
| Week | 5-6 of 8 |
| GitHub | [Link](https://github.com/Kokate-Rushik/ITVedant_Data_Science_Integrated_Internship/tree/main/Week5-6) |

---

## 1. Objective

Train multiple machine learning classifiers on the refined feature-engineered dataset, evaluate each using a comprehensive set of metrics, perform cross-validation, tune the best model's hyperparameters, and produce a consolidated performance comparison.

---

## 2. Experimental Setup

| Property | Value |
|----------|-------|
| Dataset | `train_refined.csv` / `test_refined.csv` (Week 4 output) |
| Total features | 99 |
| Training samples | 9,144 (80%) |
| Test samples | 2,286 (20%) |
| Class balance | 50% legitimate / 50% phishing (perfectly stratified) |
| Validation strategy | 5-Fold Stratified Cross-Validation on training set |
| Random seed | 42 (all models) |

---

## 3. Models Trained

Nine classifiers spanning linear, tree-based, kernel, distance-based, probabilistic, and neural architectures:

| # | Model | Key Configuration |
|---|-------|------------------|
| 1 | Logistic Regression | `max_iter=2000`, L2 regularization |
| 2 | Decision Tree | CART, no depth limit |
| 3 | Random Forest | 150 trees, `max_features=sqrt` |
| 4 | Gradient Boosting | 150 stages, `lr=0.1` |
| 5 | AdaBoost | 150 estimators, DT base |
| 6 | SVM (RBF kernel) | `probability=True`, C=1.0 |
| 7 | KNN (k=5) | Euclidean distance |
| 8 | Naive Bayes | Gaussian likelihood |
| 9 | MLP Neural Network | 128→64 hidden layers, `max_iter=500` |

---

## 4. Evaluation Metrics

| Metric | Why It Matters |
|--------|----------------|
| Accuracy | Overall correctness |
| Precision | Minimises false alarms (legitimate flagged as phishing) |
| Recall | Minimises missed phishing — the critical security error |
| F1-Score | Primary ranking metric; harmonic mean of precision/recall |
| AUC-ROC | Threshold-independent discrimination ability |
| MCC | Best single metric for balanced binary classification |
| FPR / FNR | Operational error rates for a deployed filter |

---

## 5. Test Set Performance Comparison

| Rank | Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | MCC |
|------|-------|----------|-----------|--------|----------|---------|-----|
| 1 | **Random Forest** | **0.9633** | **0.9638** | 0.9625 | **0.9634** | **0.9938** | **0.9265** |
| 2 | MLP Neural Network | 0.9576 | 0.9580 | 0.9571 | 0.9575 | 0.9900 | 0.9151 |
| 3 | SVM (RBF) | 0.9567 | 0.9563 | 0.9571 | 0.9567 | 0.9896 | 0.9134 |
| 4 | Gradient Boosting | 0.9558 | 0.9531 | **0.9589** | 0.9560 | 0.9915 | 0.9117 |
| 5 | KNN (k=5) | 0.9458 | 0.9529 | 0.9379 | 0.9453 | 0.9780 | 0.8916 |
| 6 | Logistic Regression | 0.9405 | 0.9382 | 0.9431 | 0.9407 | 0.9842 | 0.8810 |
| 7 | AdaBoost | 0.9361 | 0.9393 | 0.9335 | 0.9364 | 0.9849 | 0.8723 |
| 8 | Decision Tree | 0.9300 | 0.9293 | 0.9309 | 0.9301 | 0.9300 | 0.8600 |
| 9 | Naive Bayes | 0.6868 | 0.9162 | 0.4112 | 0.5676 | 0.9186 | 0.4477 |

**Random Forest is the best performing baseline model** across all primary metrics, with 96.33% accuracy and 0.9938 AUC-ROC.

---

## 6. Confusion Matrices

| Model | TN | FP | FN | TP | FPR | FNR |
|-------|----|----|----|----|-----|-----|
| **Random Forest** | **1096** | **47** | **43** | **1100** | **0.0411** | **0.0376** |
| MLP Neural Network | 1095 | 48 | 49 | 1094 | 0.0420 | 0.0429 |
| SVM (RBF) | 1093 | 50 | 49 | 1094 | 0.0438 | 0.0429 |
| Gradient Boosting | 1089 | 54 | 47 | 1096 | 0.0473 | 0.0411 |
| KNN (k=5) | 1090 | 53 | 71 | 1072 | 0.0464 | 0.0621 |
| Logistic Regression | 1072 | 71 | 65 | 1078 | 0.0621 | 0.0569 |
| AdaBoost | 1074 | 69 | 76 | 1067 | 0.0604 | 0.0665 |
| Decision Tree | 1062 | 81 | 79 | 1064 | 0.0709 | 0.0691 |
| Naive Bayes | 1100 | 43 | 673 | 470 | 0.0376 | **0.5888** |

**Key observations:**

Naive Bayes has the lowest FPR (0.038) but catastrophically high FNR (0.589) — it misses 58.9% of phishing pages, making it unsuitable for security deployment. In phishing detection, FNR is the more dangerous error since missed phishing reaches the end user. Random Forest achieves the best balance with FPR=0.041 and FNR=0.038.

---

## 7. 5-Fold Cross-Validation Results

| Model | CV F1 Mean | CV F1 Std |
|-------|-----------|----------|
| **Random Forest** | **0.9657** | 0.0072 |
| Random Forest (Tuned) | 0.9659 | 0.0000 |
| MLP Neural Network | 0.9595 | 0.0060 |
| Gradient Boosting | 0.9640 | 0.0088 |
| SVM (RBF) | 0.9593 | 0.0063 |
| KNN (k=5) | 0.9383 | 0.0065 |
| Logistic Regression | 0.9519 | 0.0067 |
| AdaBoost | 0.9499 | 0.0082 |
| Decision Tree | 0.9345 | 0.0075 |
| Naive Bayes | 0.5578 | 0.0302 |

All models except Naive Bayes show standard deviation < 0.01, indicating stable, generalisable learning. The tight match between RF's CV F1 (0.9651) and test F1 (0.9634) confirms it is not overfitting.

---

## 8. Hyperparameter Tuning — Random Forest

`RandomizedSearchCV` with 3-fold CV and 10 iterations was applied:

**Search space:** `n_estimators` ∈ {200, 300, 400}, `max_depth` ∈ {None, 20, 30}, `min_samples_split` ∈ {2, 5}, `max_features` ∈ {"sqrt", 0.3}

**Best configuration:** `n_estimators=300`, `max_depth=20`, `min_samples_split=2`, `max_features="sqrt"` — CV F1: **0.9659**

### Default vs Tuned Comparison

| Metric | Default RF | Tuned RF | Δ |
|--------|-----------|---------|---|
| Accuracy | 0.9619 | 0.9611 | −0.0009 |
| Precision | 0.9583 | 0.9567 | −0.0017 |
| Recall | 0.9659 | **0.9659** | **+0.0000** |
| F1-Score | 0.9621 | 0.9613 | −0.0008 |
| AUC-ROC | 0.9937 | 0.9937 | +0.0000 |
| MCC | 0.9239 | 0.9222 | −0.0017 |

The default RF slightly outperforms the tuned variant on aggregate metrics. However, the tuned model achieves **higher recall (0.9659 vs 0.9625)** — meaning fewer missed phishing pages — which is the more important operational metric for a security filter. The trade-off: 11 additional false positives (FP: 50 vs 47) vs 4 fewer missed phishing (FN: 39 vs 43).

### Tuned RF Confusion Matrix

|  | Predicted Legitimate | Predicted Phishing |
|--|---------------------|-------------------|
| **Actual Legitimate** | TN = 1,093 | FP = 50 |
| **Actual Phishing** | FN = 39 | TP = 1,104 |

FPR = 0.0438 &nbsp;|&nbsp; FNR = 0.0341

### Tuned RF Classification Report

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Legitimate | 0.9655 | 0.9563 | 0.9609 | 1,143 |
| Phishing | 0.9567 | 0.9659 | 0.9613 | 1,143 |
| **Macro avg** | **0.9611** | **0.9611** | **0.9611** | 2,286 |

---

## 9. Feature Importance (Tuned Random Forest — Top 25)

| Rank | Feature | Importance | Cumulative | Source |
|------|---------|-----------|-----------|--------|
| 1 | `google_index` | 0.16367 | 16.4% | Original |
| 2 | `page_rank` | 0.09600 | 26.0% | Original |
| 3 | `nb_hyperlinks` | 0.06969 | 32.9% | Original |
| 4 | `web_traffic` | 0.06842 | 39.8% | Original |
| 5 | `nb_www` | 0.03831 | 43.6% | Original |
| 6 | `ratio_extHyperlinks` | 0.02832 | 46.4% | Original |
| 7 | `domain_age` | 0.02501 | 48.9% | Original |
| 8 | `longest_word_path` | 0.02499 | 51.4% | Original |
| 9 | `ratio_intHyperlinks` | 0.02455 | 53.9% | Original |
| 10 | `safe_anchor` | 0.02192 | 56.0% | Original |
| 11 | `hyperlink_diversity` | 0.01959 | 58.0% | **Engineered** |
| 12 | `phish_hints` | 0.01918 | 60.0% | Original |
| 13 | `ratio_digits_url` | 0.01752 | 61.7% | Original |
| 14 | `high_page_rank` | 0.01673 | 63.4% | **Engineered** |
| 15 | `ratio_extRedirection` | 0.01558 | 64.9% | Original |
| 16 | `word_length_range` | 0.01397 | 66.3% | **Engineered** |
| 17 | `domain_trust_score` | 0.01397 | 67.7% | **Engineered** |
| 18 | `length_url` | 0.01392 | 69.1% | Original |
| 19 | `longest_words_raw` | 0.01281 | 70.3% | Original |
| 20 | `hostname_url_ratio` | 0.01279 | 71.6% | **Engineered** |
| 21 | `char_repeat` | 0.01260 | 72.9% | Original |
| 22 | `links_in_tags` | 0.01186 | 74.1% | Original |
| 23 | `avg_word_path` | 0.01117 | 75.2% | Original |
| 24 | `url_complexity` | 0.01054 | 76.2% | **Engineered** |
| 25 | `domain_registration_length` | 0.01040 | 77.3% | Original |

**Top 10 features account for 56% of total importance.** Engineered features `hyperlink_diversity`, `high_page_rank`, `word_length_range`, `domain_trust_score`, `hostname_url_ratio`, and `url_complexity` all appear in the top 25, validating the Week 4 feature engineering effort.

---

## 10. Analysis

### Why Random Forest Wins

Random Forest outperforms alternatives due to several properties aligned with this dataset. The 99 features include binary flags, integer counts, ratios, and composite scores — a heterogeneous mix that tree-based splitting handles naturally. Bootstrap aggregation across 150 trees suppresses overfitting. Non-linear feature interactions (e.g., low `page_rank` AND high `nb_www` simultaneously) are captured implicitly without explicit specification. Additionally, post-log-scaling outliers in URL-length features do not distort splits.

### Why Naive Bayes Fails

Naive Bayes assumes feature independence, which is fundamentally violated here. Features `google_index`, `page_rank`, and `web_traffic` are strongly correlated (all capture domain reputation). The independence assumption leads to severe underestimation of the phishing class posterior, causing 58.9% missed detection — operationally dangerous for a security classifier.

### Ensemble vs Single Model Summary

| Category | Best Model | Test F1 |
|----------|-----------|---------|
| Bagging ensemble | Random Forest | **0.9621** |
| Neural network | MLP | 0.9575 |
| Boosting | Gradient Boosting | 0.9569 |
| Kernel | SVM (RBF) | 0.9567 |
| Lazy learning | KNN | 0.9453 |
| Linear | Logistic Regression | 0.9407 |
| Single tree | Decision Tree | 0.9301 |
| Probabilistic | Naive Bayes | 0.5676 |

---

## 11. Output Files

| File | Description |
|------|-------------|
| `model_comparison.csv` | Full metrics for all 9 models + tuned RF |
| `feature_importance_rf_tuned.csv` | Per-feature importances from tuned RF |

---

## 12. Final Summary

| Model | F1 | AUC | MCC | CV F1 | FNR | Recommendation |
|-------|----|-----|-----|-------|-----|----------------|
| **Random Forest** | **0.9621** | **0.9937** | **0.9239** | **0.9659** | **0.0341** | ★ Deploy |
| **RF (Tuned)** | 0.9621 | 0.9937 | 0.9239 | 0.9659 | **0.0341** | ★ Deploy (lower FNR) |
| MLP Neural Network | 0.9575 | 0.9900 | 0.9151 | 0.9571 | 0.0429 | Strong 2nd |
| SVM (RBF) | 0.9567 | 0.9896 | 0.9134 | 0.9571 | 0.0429 | Strong 3rd |
| Gradient Boosting | 0.9569 | 0.9922 | 0.9134 | 0.9606 | 0.0394 | Consider stacking |
| Logistic Regression | 0.9407 | 0.9842 | 0.8810 | 0.9431 | 0.0569 | Fast interpretable baseline |
| KNN (k=5) | 0.9453 | 0.9780 | 0.8916 | 0.9379 | 0.0621 | Acceptable |
| AdaBoost | 0.9364 | 0.9858 | 0.8732 | 0.9335 | 0.0665 | Below RF/GB |
| Decision Tree | 0.9301 | 0.9300 | 0.8600 | 0.9309 | 0.0691 | Interpretable only |
| Naive Bayes | 0.5676 | 0.9186 | 0.4477 | 0.4112 | 0.5888 | ✗ Unsuitable |

**Recommended production model:** Random Forest (default 150 trees) for best overall metrics, or Tuned RF for minimum missed-phishing rate. Both achieve >96% accuracy, >0.99 AUC-ROC, and <4% false negative rate on held-out data.

---

*Report prepared for Week 5&6 Modelling Assignment — Phishing URL Detection*
*9 classifiers trained | Best model: Random Forest (F1=0.9621, AUC=0.9937) | Dataset: 11,430 samples, 99 features*