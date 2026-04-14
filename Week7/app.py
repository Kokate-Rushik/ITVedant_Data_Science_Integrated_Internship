import os
import pickle
import json
import logging
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from zoneinfo import ZoneInfo

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── App initialisation ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ── Load model ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.pkl")
_model_bundle = None

def load_model():
    """Load model from disk; cache in memory."""
    global _model_bundle
    if _model_bundle is None:
        with open(MODEL_PATH, "rb") as f:
            _model_bundle = pickle.load(f)
        logger.info(f"Model loaded: {MODEL_PATH}")
    return _model_bundle

def get_model():
    bundle = load_model()
    return bundle["model"], bundle["feature_names"], bundle["meta"]

# ── Helper: build prediction response ─────────────────────────────────────────
THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", "0.5"))

def _predict_one(features_dict, model, feature_names, meta):
    """Run prediction for a single record dict. Returns result dict."""
    missing = [f for f in feature_names if f not in features_dict]
    if missing:
        raise ValueError(f"Missing features: {missing[:5]}{'...' if len(missing)>5 else ''}")

    row = pd.DataFrame([features_dict])[feature_names]
    prob = float(model.predict_proba(row)[0, 1])
    label = "phishing" if prob >= THRESHOLD else "legitimate"
    confidence = prob if label == "phishing" else 1 - prob

    # Local attribution: fi × |value|
    fi_arr = model.feature_importances_
    raw_attr = fi_arr * np.abs(row.values[0])
    total = raw_attr.sum() if raw_attr.sum() > 0 else 1.0
    norm_attr = (raw_attr / total).tolist()
    top_idx = np.argsort(raw_attr)[::-1][:10]
    top_features = {
        feature_names[i]: round(float(norm_attr[i]), 6)
        for i in top_idx
    }

    return {
        "prediction": label,
        "phishing_probability": round(prob, 4),
        "legitimate_probability": round(1 - prob, 4),
        "confidence": round(confidence, 4),
        "threshold_used": THRESHOLD,
        "top_contributing_features": top_features
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    """Root — human-readable API info."""
    return jsonify({
        "service": "Phishing URL Detection API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoints": {
            "GET  /":               "This page",
            "GET  /health":         "Health check",
            "GET  /model/info":     "Model metadata",
            "GET  /features":       "Feature importance list",
            "POST /predict":        "Single prediction",
            "POST /predict/batch":  "Batch prediction (max 100)"
        },
        "predict_example": {
            "method": "POST",
            "url": "/predict",
            "content_type": "application/json",
            "body": {
                "google_index": 0,
                "page_rank": 0,
                "nb_hyperlinks": 5,
                "web_traffic": 0,
                "nb_www": 2,
                "...": "all 99 features required"
            }
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """Lightweight health check for Render/load balancer."""
    try:
        _, feature_names, meta = get_model()
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "n_features": len(feature_names),
            "test_f1": meta["test_metrics"]["f1"],
            "timestamp": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
        }), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 503


@app.route("/model/info", methods=["GET"])
def model_info():
    """Return model metadata: type, params, metrics, features."""
    try:
        _, feature_names, meta = get_model()
        return jsonify({
            "model_type": meta.get("model_type"),
            "n_features": meta.get("n_features"),
            "feature_names": feature_names,
            "label_map": meta.get("label_map"),
            "threshold": THRESHOLD,
            "test_metrics": meta.get("test_metrics"),
            "confusion_matrix": meta.get("confusion_matrix"),
            "top_features": meta.get("top_features")
        })
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/features", methods=["GET"])
def feature_importance():
    """Return all 99 features ranked by model importance."""
    try:
        model, feature_names, _ = get_model()
        fi = dict(sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        ))
        return jsonify({
            "n_features": len(fi),
            "feature_importances": {k: round(float(v), 6) for k, v in fi.items()}
        })
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single prediction endpoint.

    Request body (JSON):
        { "feature1": value1, "feature2": value2, ... }

    Response:
        {
            "prediction": "phishing" | "legitimate",
            "phishing_probability": float,
            "legitimate_probability": float,
            "confidence": float,
            "threshold_used": float,
            "top_contributing_features": { "feature": attribution, ... }
        }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON body"}), 400

        model, feature_names, meta = get_model()
        result = _predict_one(data, model, feature_names, meta)
        logger.info(f"Prediction: {result['prediction']} (prob={result['phishing_probability']})")
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Batch prediction endpoint.

    Request body (JSON):
        { "records": [ {feature_dict_1}, {feature_dict_2}, ... ] }
        Maximum 100 records per call.

    Response:
        {
            "n_records": int,
            "results": [ { prediction_result_1 }, ... ]
        }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if data is None or "records" not in data:
            return jsonify({"error": "Body must have key 'records': [...]"}), 400

        records = data["records"]
        if not isinstance(records, list) or len(records) == 0:
            return jsonify({"error": "'records' must be a non-empty list"}), 400
        if len(records) > 100:
            return jsonify({"error": "Maximum 100 records per batch request"}), 400

        model, feature_names, meta = get_model()
        results = []
        errors  = []
        for i, record in enumerate(records):
            try:
                res = _predict_one(record, model, feature_names, meta)
                results.append({"index": i, **res})
            except ValueError as e:
                errors.append({"index": i, "error": str(e)})

        summary = {
            "total": len(records),
            "phishing_count": sum(1 for r in results if r.get("prediction") == "phishing"),
            "legitimate_count": sum(1 for r in results if r.get("prediction") == "legitimate"),
            "error_count": len(errors)
        }

        logger.info(f"Batch: {summary}")
        return jsonify({
            "n_records": len(records),
            "summary": summary,
            "results": results,
            "errors": errors
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500


# ── Error handlers ─────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "available": [
        "/", "/health", "/model/info", "/features",
        "/predict", "/predict/batch"
    ]}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting Flask app on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)