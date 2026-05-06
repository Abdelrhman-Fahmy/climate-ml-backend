from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
import time

app = Flask(__name__)

MODEL_FILE = 'model.pkl'
model = None

# Load model
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
else:
    print("Warning: Model not found. Run train_model.py first.")

# In-memory broker state
latest_data = {
    "temp_avg": 0, "temp_dht": 0, "temp_ds": 0,
    "humidity": 0, "water_level": 0, "pir_detected": False,
    "occupancy": False, "fan_on": False, "pump_on": False,
    "cooldown_active": False, "delta_temp": 0.0,
    "score": 0.0, "effective_score": 0.0,
    "voted_fan_on": False, "dominant_factor": 0, "reason_code": 0,
    "anomaly_flag": False, "failsafe_level": 0, "system_health": 1.0,
    "system_unstable": False, "confidence": 0.0, "ml_reasoning": "Waiting for data...",
    "is_hlg": False, "humidity_estimated": False,
    "target_temp": 22.0, "target_low": 21.0, "target_high": 23.0,
    "uptime_s": 0, "timestamp": 0, "dht_contributed": False,
    "ds_contributed": False, "current_valid": True, "cmd_ack_status": 0
}
history = []
HISTORY_LIMIT = 60

pending_control = {}
pending_config = {}

@app.route('/predict', methods=['POST'])
def predict():
    global latest_data, history, pending_control, pending_config
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json or {}
    try:
        # Extract features for ML
        temp = float(data.get('temp_avg', 0))
        humidity = float(data.get('humidity', 0))
        occupancy = float(data.get('occupancy', 0) or data.get('pir_detected', 0))
        delta_temp = float(data.get('delta_temp', 0))
        rolling_avg_temp = float(data.get('rolling_avg_temp', temp)) # Optional from ESP
        prev_fan_state = float(data.get('fan_on', 0))

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

        # Update latest state with all incoming fields
        latest_data.update(data)
        latest_data["confidence"] = confidence
        latest_data["ml_reasoning"] = reason_str
        latest_data["reason_code"] = int(prediction) # Syncing reason code with prediction
        latest_data["timestamp"] = int(time.time())

        # Update history
        history.append(latest_data.copy())
        if len(history) > HISTORY_LIMIT:
            history.pop(0)

        # Prepare response to ESP32
        response_payload = {
            "fan_state": int(prediction),
            "confidence": confidence,
            "reasoning": reason_str,
            "contributions": contributions,
            "pending_control": pending_control.copy(),
            "pending_config": pending_config.copy()
        }

        # Clear pending queues
        pending_control.clear()
        pending_config.clear()

        return jsonify(response_payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/data', methods=['GET'])
def get_data():
    return jsonify(latest_data)

@app.route('/control', methods=['POST'])
def control():
    global pending_control
    data = request.json or {}
    # Expected from Flutter: {'fan': bool, 'pump': bool, 'cmd_id': int}
    if 'fan' in data:
        pending_control['fan'] = bool(data['fan'])
    if 'pump' in data:
        pending_control['pump'] = bool(data['pump'])
    if 'cmd_id' in data:
        pending_control['cmd_id'] = int(data['cmd_id'])
    return jsonify({"ok": True, "msg": "Command queued for ESP32"})

@app.route('/config', methods=['POST'])
def update_config():
    global pending_config
    data = request.json or {}
    pending_config.update(data)
    return jsonify({"ok": True, "msg": "Config queued for ESP32"})

@app.route('/history', methods=['GET'])
def get_history():
    limit = request.args.get('limit', HISTORY_LIMIT, type=int)
    return jsonify(history[-limit:])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
