from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

MODEL_FILE = 'model.pkl'
model = None

# Load model
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
else:
    print("Warning: Model not found. Run train_model.py first.")

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    try:
        # Extract features
        temp = float(data.get('temp', 0))
        humidity = float(data.get('humidity', 0))
        occupancy = float(data.get('occupancy', 0))
        delta_temp = float(data.get('delta_temp', 0))
        rolling_avg_temp = float(data.get('rolling_avg_temp', 0))
        prev_fan_state = float(data.get('prev_fan_state', 0))

        features = np.array([[temp, humidity, occupancy, delta_temp, rolling_avg_temp, prev_fan_state]])
        
        # Predict
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[1] if prediction == 1 else probabilities[0])

        # Explainability logic
        weights = model.coef_[0]
        contributions = {
            "temp": float(weights[0] * temp),
            "humidity": float(weights[1] * humidity),
            "occupancy": float(weights[2] * occupancy),
            "delta_temp": float(weights[3] * delta_temp),
            "rolling_avg_temp": float(weights[4] * rolling_avg_temp),
            "prev_fan_state": float(weights[5] * prev_fan_state)
        }
        
        # Determine top reason
        reasoning = []
        if prediction == 1:
            if contributions["temp"] > 1.0:
                reasoning.append("high temperature")
            if contributions["delta_temp"] > 0.5:
                reasoning.append("temperature is rising rapidly")
            if contributions["occupancy"] > 0.5:
                reasoning.append("occupancy detected")
            if not reasoning:
                reasoning.append("learned trends indicate necessity")
            reason_str = f"Fan ON because {' and '.join(reasoning)}."
        else:
            reason_str = "Fan OFF because temperature is stable and acceptable."

        return jsonify({
            "fan_state": int(prediction),
            "confidence": confidence,
            "reasoning": reason_str,
            "contributions": contributions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
