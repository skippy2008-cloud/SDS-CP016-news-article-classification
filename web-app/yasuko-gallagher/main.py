# Import Streamlit library
import streamlit as st
import joblib
import pickle
import re
import nltk
import sys

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def clean_text(text):
    review = text
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    
    return review

# Load the trained model 
model_path = 'classifier_BOW.pkl'  
try:
    classifier = joblib.load(model_path)
except FileNotFoundError:
    print("Error: Model file not found. Please check the path.")
    sys.exit(1)

# Function to predict using the classifier
def predict_text(text):
    # Preprocess the text
    processed_text = clean_text(text)

    prediction = classifier.predict([processed_text])  # Pass a list if required
    return prediction


# Main program
if __name__ == "__main__":
    # Get user input
    user_input = input("Enter text to classify: ")
    
    # Predict
    result = predict_text(user_input)
    
    # Output the prediction
    print(f"The prediction for the input is: {result}")




