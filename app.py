from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained models
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("sentiment_model.pkl")

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Transform input text using TF-IDF
    text_transformed = vectorizer.transform([text])

    # Predict sentiment
    # Convert numeric prediction to human-readable sentiment
    sentiment_mapping = {0: "negative", 1: "neutral", 2: "positive"}
    prediction = int(model.predict(text_transformed)[0])
    predicted_sentiment = sentiment_mapping.get(prediction, "unknown")



    return jsonify({"text": text, "predicted_sentiment": predicted_sentiment})

if __name__ == '__main__':
    app.run(debug=True)
