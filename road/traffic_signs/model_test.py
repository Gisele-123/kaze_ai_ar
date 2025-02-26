import joblib
import cv2
import numpy as np

# Load the trained model
model = joblib.load("traffic_sign_model.pkl")

# Define image size
IMG_SIZE = (64, 64)

# Load and preprocess a test image
img_path = "TEST/049_0021_j.png"  # Change to a real image path
img = cv2.imread(img_path)
if img is None:
    print("Error: Image not found.")
else:
    img = cv2.resize(img, IMG_SIZE).flatten().reshape(1, -1)

    # Predict the class
    prediction = model.predict(img)
    print(f"Predicted Class ID: {prediction[0]}")
