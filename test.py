import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image
import os
import argparse

def load_model(model_path):
    """Load the pretrained Keras model."""
    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess the image to match the model's expected input.
    Adjust target_size according to your model's requirements.
    """
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = np.array(img, dtype = np.float32) / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_spoof(model, image_array):
    """Make prediction using the model."""
    try:
        prediction = model.predict(image_array)
        return prediction
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def interpret_result(prediction, threshold=0.5):
    """Interpret the model's prediction."""
    try:
        # For binary classification, we typically have a single output
        score = prediction[0][0]
        result = "Real" if score < threshold else "Spoof"
        confidence = score if result == "Spoof" else 1 - score
        return result, confidence
    except Exception as e:
        print(f"Error interpreting result: {e}")
        return None, None

def main():
    # Load model
    model = load_model('model/best_model_v2.keras')
    if model is None:
        return
    
    # Preprocess image
    img_array = preprocess_image('image_test/fake_1.jpeg')
    if img_array is None:
        return
    
    # Make prediction
    prediction = predict_spoof(model, img_array)
    if prediction is None:
        return
    
    # Interpret results
    result, confidence = interpret_result(prediction, threshold=0.5)
    
    # Display results
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()