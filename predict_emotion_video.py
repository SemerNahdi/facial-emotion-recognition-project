import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Define paths
MODEL_DIR = 'C:\\Users\\ASUS\\Desktop\\ING3\\ComputerVision\\Facial Emotion Recognition\\models'
FINE_TUNED_MODEL_PATH = os.path.join(MODEL_DIR, 'fine_tuned_model.keras')
CATEGORIES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = 128  # Ensure this matches your model's input size

def load_and_preprocess_image(image):
    """
    Preprocesses an image for prediction.
    Args:
        image (numpy array): The image loaded or frame captured from the camera.
    Returns:
        numpy array: Preprocessed image ready for prediction.
    """
    try:
        # Resize the image to IMG_SIZE x IMG_SIZE
        img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
        # Normalize pixel values to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Expand dimensions to match model's input shape (1, IMG_SIZE, IMG_SIZE, 3)
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

def predict_emotion(image, model):
    """
    Predicts the emotion from the given image or frame.
    Args:
        image (numpy array): The image or frame to process.
        model (Keras model): Loaded Keras model.
    Returns:
        str: Predicted emotion label.
    """
    # Preprocess the image
    preprocessed_image = load_and_preprocess_image(image)
    
    if preprocessed_image is None:
        return "Error during preprocessing, could not predict."
    
    # Make prediction
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Get the corresponding emotion label
    predicted_emotion = CATEGORIES[predicted_class]
    
    return predicted_emotion

def open_webcam():
    """
    Opens the webcam, detects faces in real-time, and predicts emotion for each face.
    """
    # Load the model
    model = load_model(FINE_TUNED_MODEL_PATH)
    
    # Start webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break
        
        # Use only the face region for emotion prediction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # If a face is detected, predict emotion
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]  # Crop the face
            emotion = predict_emotion(face, model)
            
            # Display the predicted emotion on the frame
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Emotion Recognition', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

def upload_photo():
    """
    Allows the user to upload a photo, detects the face, and predicts the emotion.
    """
    # Ask the user to select an image file
    file_path = filedialog.askopenfilename()

    if not file_path:
        print("No file selected.")
        return
    
    # Load the image
    img = cv2.imread(file_path)
    if img is None:
        print(f"Error: Could not load image at {file_path}")
        return
    
    # Load the model
    model = load_model(FINE_TUNED_MODEL_PATH)

    # Use only the face region for emotion prediction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("No faces detected.")
        return

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]  # Crop the face
        emotion = predict_emotion(face, model)

        # Draw the predicted emotion on the image
        cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Show the image with emotion labels
    cv2.imshow("Emotion Recognition", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_gui():
    """
    Creates a simple GUI with options to either upload a photo or open the webcam.
    """
    # Create the main window
    root = tk.Tk()
    root.title("Facial Emotion Recognition")

    # Create and position buttons
    btn_photo = tk.Button(root, text="Upload Photo", command=upload_photo, width=30, height=2)
    btn_photo.pack(pady=20)

    btn_webcam = tk.Button(root, text="Open Webcam", command=open_webcam, width=30, height=2)
    btn_webcam.pack(pady=20)

    # Run the main loop
    root.mainloop()

if __name__ == "__main__":
    create_gui()


