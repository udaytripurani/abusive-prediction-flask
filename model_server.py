from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS for handling cross-origin requests
import joblib
import numpy as np
import string
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK stopwords if you haven't already
nltk.download('stopwords')

app = Flask(__name__)

# Apply CORS and allow requests from all origins (everywhere)
CORS(app)  # Allow all origins

# Load the LightGBM model and the TF-IDF vectorizer
model = joblib.load('lgbm_abusive_model.pkl')  # Load the trained LightGBM model
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Load the TF-IDF vectorizer

# Function to clean and preprocess text
def clean_text(text):
    """
    This function takes raw text as input and performs the following preprocessing steps:
    1. Removes punctuation
    2. Converts the text to lowercase
    3. Removes stopwords using NLTK stopwords
    """
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

@app.route('/predict', methods=['POST'])
def predict():
    """
    This function handles POST requests to the /predict route.
    It expects JSON input with a 'text' field, preprocesses the text,
    vectorizes it, and then uses the pre-trained model to predict
    whether the text is abusive or not.
    """
    try:
        # Get the input data from the request body
        data = request.get_json()

        # Check if 'text' is present in the input data
        if 'text' not in data:
            return jsonify({'error': "'text' field is required"}), 400

        # Extract the text from the incoming JSON
        text = data['text']

        # Clean and preprocess the input text
        cleaned_text = clean_text(text)

        # Vectorize the cleaned text using the pre-loaded TF-IDF vectorizer
        text_vectorized = vectorizer.transform([cleaned_text])

        # Predict using the pre-trained LightGBM model
        prediction = model.predict(text_vectorized)

        # Return the prediction result as a JSON response
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        # Return an error response if something goes wrong
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app on port 5000, accessible publicly
    app.run(debug=True, host='0.0.0.0', port=5000)
