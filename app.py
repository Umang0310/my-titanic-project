import logging
import pandas as pd
from flask import Flask, request, jsonify
import joblib

# 1. Initialize Flask App
app = Flask(__name__)

# 2. Configure Logging
# This will create a file named 'api.log' to store request info
logging.basicConfig(
    filename='api.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# 3. Load the Serialized Model
try:
    model = joblib.load('titanic_model.pkl')
    app.logger.info("Model loaded successfully from 'titanic_model.pkl'")
except FileNotFoundError:
    app.logger.error("Model file 'titanic_model.pkl' not found.")
    model = None
except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    model = None

# 4. Define a Root/Health-Check Endpoint
@app.route('/', methods=['GET'])
def health_check():
    """A simple health check endpoint."""
    return "Flask API is running!"

# 5. Define the Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to make a prediction."""
    if not model:
        return jsonify({"error": "Model is not loaded."}), 500

    try:
        # Get data from the POST request
        data = request.get_json()
        app.logger.info(f"Received prediction request: {data}")

        # Convert JSON data to a pandas DataFrame
        features_df = pd.DataFrame([data])
        
        # --- IMPORTANT ---
        # Ensure correct column order and names (all lowercase)
        # to match your 'train.py' script.
        features_df = features_df[['pclass', 'sex', 'age', 'fare', 'embarked']]

        # Make prediction
        prediction = model.predict(features_df)
        probability = model.predict_proba(features_df)

        # Format the output
        survived_bool = bool(prediction[0])
        survived_proba = probability[0][1] # Probability of 'Survived'

        output = {
            'prediction': 'Survived' if survived_bool else 'Did not survive',
            'confidence_score': f"{survived_proba:.4f}"
        }

        # Log the prediction
        app.logger.info(f"Sending prediction: {output}")
        return jsonify(output)

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400

# 6. Run the App
if __name__ == '__main__':
    # This runs the app in development mode
    app.run(debug=True, port=5000)