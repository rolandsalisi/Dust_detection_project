from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API from any origin

# ── Load models ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

kmeans         = joblib.load(os.path.join(BASE, "models", "kmeans_dust_model.pkl"))
scaler         = joblib.load(os.path.join(BASE, "models", "scaler.pkl"))
features       = joblib.load(os.path.join(BASE, "models", "features.pkl"))
cluster_labels = joblib.load(os.path.join(BASE, "models", "cluster_labels.pkl"))

print("✅ Models loaded")
print("   Features      :", features)
print("   Cluster labels:", cluster_labels)


# ── Helper: build feature vector from rolling window ─────────
def compute_features(dust_window: list, temp_window: list) -> dict:
    dust = np.array(dust_window)
    temp = np.array(temp_window)
    return {
        "Dust_Density_mg_per_m3": float(dust[-1]),
        "Temp_MA_5":              float(np.mean(temp)),
        "Dust_Temp_Product":      float(dust[-1] * temp[-1]),
        "Temp_Median_5":          float(np.median(temp)),
        "Dust_Max_5":             float(np.max(dust)),
        "Temp_Max_5":             float(np.max(temp)),
    }


# ── POST /predict ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
    {
        "dust_window": [0.05, 0.04, 0.06, 0.05, 0.07],   // last 5 dust readings
        "temp_window": [32.1, 33.0, 32.5, 32.8, 33.2]    // last 5 temp readings
    }
    Returns:
    {
        "label":    "High Dust",
        "cluster":  1,
        "features": { ... },
        "status":   "ok"
    }
    """
    data = request.get_json(force=True)

    dust_window = data.get("dust_window", [])
    temp_window = data.get("temp_window", [])

    WINDOW = 5
    if len(dust_window) < WINDOW or len(temp_window) < WINDOW:
        return jsonify({
            "label":   "Collecting data...",
            "cluster": None,
            "status":  "collecting",
            "message": f"Need {WINDOW} readings, got {len(dust_window)}"
        })

    # Use only the last WINDOW readings
    dust_window = dust_window[-WINDOW:]
    temp_window = temp_window[-WINDOW:]

    # Feature engineering (mirrors Python pipeline exactly)
    feature_dict = compute_features(dust_window, temp_window)

    # Arrange in correct order from features.pkl
    X = np.array([[feature_dict[f] for f in features]])

    # Scale
    X_scaled = scaler.transform(X)

    # Predict cluster
    cluster = int(kmeans.predict(X_scaled)[0])
    label   = cluster_labels[cluster]

    return jsonify({
        "label":    label,
        "cluster":  cluster,
        "features": feature_dict,
        "status":   "ok"
    })


# ── GET /health ───────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":   "ok",
        "models":   ["kmeans_dust_model", "scaler", "features", "cluster_labels"],
        "features": features,
        "clusters": {str(k): v for k, v in cluster_labels.items()}
    })


# ── GET /latest-from-firebase ────────────────────────────────
# Optional: backend fetches Firebase and returns prediction in one call
@app.route("/latest", methods=["GET"])
def latest():
    """
    Fetch latest sensor reading from Firebase REST API and return it.
    Frontend can call this instead of Firebase directly.
    """
    import urllib.request, json as _json

    FB_URL  = "https://mor-backend-781ab-default-rtdb.asia-southeast1.firebasedatabase.app"
    FB_AUTH = "dHg0TauW07ihA0pzCRhQwrouvRhz4ZdO3mRiecJG"

    try:
        url = f"{FB_URL}/sensor_data/latest.json?auth={FB_AUTH}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            sensor = _json.loads(resp.read())
        return jsonify({"status": "ok", "data": sensor})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
