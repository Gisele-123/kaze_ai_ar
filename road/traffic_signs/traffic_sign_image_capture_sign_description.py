import cv2
import joblib
import numpy as np
import pandas as pd
from deep_translator import GoogleTranslator

model = joblib.load("traffic_sign_model.pkl")

IMG_SIZE = (64, 64)


LABELS_FILE = "labels.csv"
labels_df = pd.read_csv(LABELS_FILE)
class_labels = {row["ClassId"]: row["Description"] for _, row in labels_df.iterrows()}

def translate_text(text, target_lang="en"):
    """Translate text into the selected language (en = English, fr = French, sw = Swahili)."""
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as e:
        print(f"Translation Error: {e}")
        return text  

def capture_and_predict():
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'SPACE' to capture an image or 'ESC' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        cv2.imshow("Traffic Sign Detection", frame)

        key = cv2.waitKey(1)
        if key == 27:  
            break
        elif key == 32:  
            cap.release()
            cv2.destroyAllWindows()

            img = cv2.resize(frame, IMG_SIZE).flatten().reshape(1, -1)

            prediction = model.predict(img)
            predicted_class_id = prediction[0]
            
            traffic_sign_description = class_labels.get(predicted_class_id, "Unknown Sign")

            print("\nChoose a language for the description:")
            print("1. English (en)")
            print("2. French (fr)")
            print("3. Swahili (sw)")
            lang_choice = input("Enter your choice (en/fr/sw): ").strip().lower()

            if lang_choice not in ["en", "fr", "sw"]:
                print("Invalid choice, defaulting to English.")
                lang_choice = "en"

            translated_description = translate_text(traffic_sign_description, lang_choice)

            print(f"\nPredicted Class ID: {predicted_class_id}")
            print(f"Traffic Sign: {traffic_sign_description}")
            print(f"Translated: {translated_description}")

            return 

    cap.release()
    cv2.destroyAllWindows()

capture_and_predict()
