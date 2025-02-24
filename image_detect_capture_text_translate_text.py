import cv2
import pytesseract
from PIL import Image
import os
from deep_translator import GoogleTranslator

pytesseract.pytesseract.tesseract_cmd = r'c:\Program Files\Tesseract-OCR\tesseract.exe'

def capture_image():
    """
    Opens webcam and allows user to capture an image.
    Returns the path of the saved image.
    """
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return None

    print("Press 's' to capture image or 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        cv2.imshow("Press 's' to Capture Image", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # Save image on pressing 's'
            image_path = "captured_image.jpg"
            cv2.imwrite(image_path, frame)
            print("\n📸 Image captured successfully!")
            cap.release()
            cv2.destroyAllWindows()
            return image_path
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

def extract_text_from_image(image_path):
    """
    Extracts text from a given image using Tesseract OCR.
    Returns the extracted text.
    """
    try:
        image = Image.open(image_path)
        
        image = image.resize((image.width * 2, image.height * 2)) 
        image = image.convert("L") 
        
        extracted_text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
        extracted_text = extracted_text.strip()

        return extracted_text if extracted_text else "⚠️ No readable text found."
    except Exception as e:
        print(f"Error: {e}")
        return None

def translate_text(text, target_language):
    """
    Translates the extracted text into the target language.
    Supported languages: English, French, Swahili.
    """
    lang_map = {"english": "en", "french": "fr", "swahili": "sw"}

    if target_language.lower() not in lang_map:
        print("❌ Invalid language selection! Defaulting to English.")
        target_language = "english"

    translated_text = GoogleTranslator(source="auto", target=lang_map[target_language.lower()]).translate(text)
    
    return translated_text

if __name__ == "__main__":
    image_path = capture_image()
    
    if image_path:
        extracted_text = extract_text_from_image(image_path)
        print("\n📝 Extracted Text:\n", extracted_text)

        if extracted_text and extracted_text != "⚠️ No readable text found.":
            target_language = input("\nTranslate to (English/French/Swahili): ").strip().lower()
            translated_text = translate_text(extracted_text, target_language)
            
            print(f"\n🌍 Translated Text ({target_language.capitalize()}):\n", translated_text)

        os.remove(image_path)
