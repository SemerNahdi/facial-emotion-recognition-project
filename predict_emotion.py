import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Define paths
MODEL_DIR = 'C:\\Users\\ASUS\\Desktop\\ING3\\ComputerVision\\Facial Emotion Recognition\\models'
FINE_TUNED_MODEL_PATH = os.path.join(MODEL_DIR, 'fine_tuned_model.keras') 
CATEGORIES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = 128  # Make sure this matches your model's input size

def load_and_preprocess_image(image_path):
    """
    Loads and preprocesses an image for prediction.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy array: Preprocessed image ready for prediction.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image at {image_path}")
        
        # Resize the image to IMG_SIZE x IMG_SIZE
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Normalize pixel values to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Expand dimensions to match model's input shape (1, IMG_SIZE, IMG_SIZE, 3)
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

def predict_emotion(image_path, model):
    """
    Predicts the emotion of the given image.

    Args:
        image_path (str): Path to the image file.
        model (Keras model): Loaded Keras model.

    Returns:
        str: Predicted emotion label.
    """
    # Preprocess the image
    preprocessed_image = load_and_preprocess_image(image_path)
    
    if preprocessed_image is None:
        return "Error during preprocessing, could not predict."
    
    # Make prediction
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Get the corresponding emotion label
    predicted_emotion = CATEGORIES[predicted_class]
    
    return predicted_emotion

def main():
    # Load the model
    if not os.path.exists(FINE_TUNED_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {FINE_TUNED_MODEL_PATH}")
    
    model = load_model(FINE_TUNED_MODEL_PATH)
    print("Model loaded successfully.")
    
    # Specify the image path
    image_path = input("Enter the path to the image you want to predict: ").strip()
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    # Predict emotion
    emotion = predict_emotion(image_path, model)
    
    if emotion:
        print(f"Predicted Emotion: {emotion}")
    else:
        print("Prediction failed.")

if __name__ == "__main__":
    main()
