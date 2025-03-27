from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import logging

# Initialize Flask app
app = Flask(__name__)

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)

# Load the trained model
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None  # Handle cases where the model cannot be loaded

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None  # Default: No prediction

    if request.method == "POST":
        temperature = request.form.get("temperature")  # Corrected name
        
        if temperature:  # Ensure input is not empty
            try:
                temperature = float(temperature)  # Convert input to float
                if model:
                    prediction = model.predict(np.array([[temperature]]))[0]
                    prediction = f"{prediction:,.2f}"  # Format prediction
                else:
                    prediction = "Model not available."
            except ValueError:
                prediction = "Invalid input. Please enter a numeric value."
        else:
            prediction = "Please enter a valid temperature."

    return render_template("index.html", prediction=prediction)

@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint for making predictions via JSON request.
    """
    data = request.get_json()

    if not data or "size" not in data:
        return jsonify({"error": "Invalid input. Please provide a 'size' value."}), 400

    try:
        size = float(data["size"])

        if model:
            predicted_price = model.predict(np.array([[size]]))[0]
            response = jsonify({"predicted_price": f"${predicted_price:,.2f}"})
            response.headers["X-Content-Type-Options"] = "nosniff"  # Security header
            return response
        else:
            return jsonify({"error": "Model not available"}), 500
    except ValueError:
        return jsonify({"error": "Invalid input. Please enter a numeric value."}), 400

if __name__ == "__main__":
    app.run(debug=True)
