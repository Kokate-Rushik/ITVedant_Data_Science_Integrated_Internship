"""
Week 6 — Part 1: Model Generation & SHAP-Equivalent Analysis
Trains the final Random Forest model, saves model.pkl,
and performs global + local interpretability analysis.

Usage:
    python generate_model.py

Outputs:
    model.pkl                  - Serialised model + metadata
    model_meta.json            - Model metadata (features, metrics)
    shap_analysis.csv          - Global feature importances
    local_explanations.json    - Per-instance feature attributions (6 samples)
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score,
                              matthews_corrcoef, confusion_matrix,
                              classification_report)

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)
df_train = pd.read_csv("./Week4/train_refined.csv")
df_test  = pd.read_csv("./Week4/test_refined.csv")

X_train = df_train.drop(columns=["status"])
y_train = df_train["status"]
X_test  = df_test.drop(columns=["status"])
y_test  = df_test["status"]

print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── 2. Train final model ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: TRAINING RANDOM FOREST (BEST MODEL)")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)
mcc  = matthews_corrcoef(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  AUC-ROC   : {auc:.4f}")
print(f"  MCC       : {mcc:.4f}")
print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

# ── 3. Global SHAP-Equivalent Analysis ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: GLOBAL FEATURE IMPORTANCE (SHAP-EQUIVALENT)")
print("=" * 60)

feature_names = X_train.columns.tolist()

# RF Mean Decrease in Impurity (MDI) — equivalent to SHAP TreeExplainer global
fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
fi.to_csv("./Week7/shap_analysis.csv", header=["importance"])
print(f"  Saved: shap_analysis.csv ({len(fi)} features)")

print(f"\n  {'Rank':<5} {'Feature':<35} {'Importance':>12} {'Cumulative':>12}")
print("  " + "-" * 68)
cumsum = 0
for rank, (feat, imp) in enumerate(fi.head(25).items(), 1):
    cumsum += imp
    print(f"  {rank:<5} {feat:<35} {imp:>12.5f} {cumsum:>11.1%}")

print(f"\n  Top 10 features explain: {fi.head(10).sum():.1%} of model decisions")

# ── 4. Local SHAP-Equivalent Explanations ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: LOCAL EXPLANATIONS (PER-INSTANCE FEATURE ATTRIBUTION)")
print("=" * 60)
print("  Method: Feature Importance × |Feature Value| attribution")
print("  (Approximates SHAP TreeExplainer local values without SHAP library)\n")

def local_explain(X_sample, model, feature_names, top_n=10):
    """
    Approximate per-instance feature attribution.
    Attribution_i = feature_importance_i × |feature_value_i|
    Normalised so attributions sum to 1.
    """
    fi_arr = model.feature_importances_
    sample_vals = X_sample.values[0]
    raw_attr = fi_arr * np.abs(sample_vals)
    total = raw_attr.sum() if raw_attr.sum() > 0 else 1.0
    normalised = raw_attr / total
    attrs = pd.Series(normalised, index=feature_names).sort_values(ascending=False)
    prob = model.predict_proba(X_sample)[0, 1]
    pred = "phishing" if prob >= 0.5 else "legitimate"
    return attrs.head(top_n), prob, pred

local_results = {}
phish_idx = X_test[y_test == 1].index[:3]
legit_idx  = X_test[y_test == 0].index[:3]

for label_type, indices in [("phishing", phish_idx), ("legitimate", legit_idx)]:
    print(f"  --- {label_type.upper()} SAMPLES ---")
    for i, idx in enumerate(indices, 1):
        sample = X_test.loc[[idx]]
        attrs, prob, pred = local_explain(sample, model, feature_names)
        key = f"{label_type}_sample_{i}"
        local_results[key] = {
            "predicted_class": pred,
            "phishing_probability": round(float(prob), 4),
            "actual_class": label_type,
            "top_features": {f: round(float(v), 6) for f, v in attrs.items()}
        }
        print(f"\n  Sample {label_type[0].upper()}{i} | Prob(phishing)={prob:.4f} | Predicted: {pred.upper()}")
        print(f"  {'Feature':<35} {'Attribution':>12}  {'Value':>10}")
        print("  " + "-" * 60)
        for feat, attr_val in attrs.items():
            raw_val = float(sample[feat].values[0])
            print(f"  {feat:<35} {attr_val:>12.5f}  {raw_val:>10.4f}")

with open("./Week7/model/local_explanations.json", "w") as f:
    json.dump(local_results, f, indent=2)
print(f"\n  Saved: local_explanations.json")

# ── 5. SHAP Summary Statistics ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: INTERPRETABILITY INSIGHTS SUMMARY")
print("=" * 60)

top5 = fi.head(5)
print("\n  TOP 5 GLOBAL PREDICTORS:")
for i, (feat, imp) in enumerate(top5.items(), 1):
    print(f"    {i}. {feat:<35} importance={imp:.5f}")

print("\n  FEATURE GROUPS BY CATEGORY:")
groups = {
    "Reputation/External": ["google_index","page_rank","web_traffic","dns_record",
                             "whois_registered_domain","domain_age","domain_registration_length"],
    "Hyperlink Structure": ["nb_hyperlinks","ratio_extHyperlinks","ratio_intHyperlinks",
                             "safe_anchor","links_in_tags","ratio_extRedirection"],
    "URL Lexical":         ["length_url","nb_www","phish_hints","ratio_digits_url",
                             "length_hostname","nb_dots","nb_hyphens"],
    "Engineered":          ["high_page_rank","domain_trust_score","hyperlink_diversity",
                             "word_length_range","hostname_url_ratio","url_complexity"],
    "Word/Token":          ["longest_word_path","longest_words_raw","avg_word_path",
                             "char_repeat","shortest_word_host"]
}
for group, feats in groups.items():
    group_imp = sum(fi.get(f, 0) for f in feats)
    print(f"    {group:<25}: {group_imp:.1%} of total importance")

# ── 6. Save model artifacts ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: SAVING MODEL ARTIFACTS")
print("=" * 60)

meta = {
    "feature_names": feature_names,
    "n_features": len(feature_names),
    "model_type": "RandomForestClassifier",
    "model_params": {k: str(v) for k, v in model.get_params().items()},
    "test_metrics": {
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4),
        "auc_roc": round(auc, 4), "mcc": round(mcc, 4)
    },
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "top_features": {k: round(float(v), 6) for k, v in fi.head(25).items()},
    "label_map": {"0": "legitimate", "1": "phishing"},
    "threshold": 0.5,
    "expected_input_columns": feature_names
}

with open("./Week7/model/model.pkl", "wb") as f:
    pickle.dump({"model": model, "feature_names": feature_names, "meta": meta}, f)
print("  Saved: model.pkl")

with open("./Week7/model/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("  Saved: model_meta.json")

print("\n" + "=" * 60)
print("MODEL GENERATION COMPLETE")
print("=" * 60)
print(f"  model.pkl              — deploy with app.py")
print(f"  model_meta.json        — API metadata reference")
print(f"  shap_analysis.csv      — global feature importances")
print(f"  local_explanations.json — sample prediction explanations")