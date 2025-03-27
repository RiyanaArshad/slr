from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Handle cases where the model cannot be loaded

@app.route("/", methods=["GET", "POST"])
def home():
    """
    Renders the homepage and handles form submissions for predictions.
    """
    prediction = None  # Default: No prediction

    if request.method == "POST":
        try:
            # Get user input
            size = float(request.form["size"])
            
            # Ensure model is loaded before prediction
            if model:
                prediction = model.predict(np.array([[size]]))[0]  # Make prediction
                prediction = f"${prediction:,.2f}"  # Format prediction to look professional
            else:
                prediction = "Model not available. Please try again later."
        except ValueError:
            prediction = "Invalid input. Please enter a valid number."

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
            return jsonify({"predicted_price": f"${predicted_price:,.2f}"})
        else:
            return jsonify({"error": "Model not available"}), 500
    except ValueError:
        return jsonify({"error": "Invalid input. Please enter a numeric value."}), 400

if __name__ == "__main__":
    app.run(debug=True)
