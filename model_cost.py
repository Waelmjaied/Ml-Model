from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

# Load the model
MODEL_PATH = './model_cost.pkl'

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"[ERROR] Could not load model from '{MODEL_PATH}': {e}")

# ✅ GET method for basic status check or sample request template
@app.route('/api/project/predict_cost', methods=['GET'])
def get_predict_cost():
    return jsonify({
        'message': 'Send a POST request with task_complexity, team_size, effective_hours, experience',
        'example_payload': {
            'task_complexity': 3,
            'team_size': 5,
            'effective_hours': 12.5,
            'experience': 2
        }
    })

# ✅ POST method for prediction
@app.route('/api/project/predict_cost', methods=['POST'])
def post_predict_cost():
    try:
        data = request.get_json()

        # Extract and validate input
        task_complexity = int(data.get('task_complexity', 0))
        team_size = int(data.get('team_size', 0))
        effective_hours = float(data.get('effective_hours', 0))
        experience = int(data.get('experience', 0))

        # Format into DataFrame
        X = pd.DataFrame([[
            task_complexity,
            team_size,
            effective_hours,
            experience
        ]])

        # Predict
        prediction = model.predict(X)[0]

        return jsonify({
            'success': True,
            'predicted_cost': round(prediction, 2)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })
    if not model:
        return jsonify({
            'success': False,
            'error': 'ML model not loaded. Check file path or permissions.'
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
