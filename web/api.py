import joblib
import os
import pandas as pd
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST


MODEL_PATH = "models/model.pkl"

WEB_PREDICTION_COUNTER = Counter(
    "web_prediction_requests_total",
    "Number of prediction requests made from the web UI"
)


def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    print("Error: 'model.pkl' no encontrado. Por favor, asegúrate de haber ejecutado el script de entrenamiento.")
    model = None

app = Flask(__name__)

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    WEB_PREDICTION_COUNTER.inc()
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        df["credit_per_month"] = df["credit_amount"] / df["duration"]

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0, 1]

        return jsonify({
            "default_prediction": int(prediction),
            "default_probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("Starting Credit Risk API on port 5000...")
    app.run(host="0.0.0.0", port=5000)
