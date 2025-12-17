import joblib
import pandas as pd
from flask import Flask, request, jsonify


MODEL_PATH = "model.pkl"

app = Flask(__name__)


def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

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
