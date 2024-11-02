import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Define paths
MODEL_DIR = 'C:\\Users\\ASUS\\Desktop\\ING3\\ComputerVision\\Facial Emotion Recognition\\models'
FINE_TUNED_MODEL_PATH = os.path.join(MODEL_DIR, 'fine_tuned_model.keras')
CATEGORIES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = 128  # Ensure this matches your model's input size

def load_and_preprocess_image(image):
    """Preprocesses an image for prediction."""
    try:
        img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

def predict_emotion(image, model):
    """Predicts the emotion from the given image."""
    preprocessed_image = load_and_preprocess_image(image)
    
    if preprocessed_image is None:
        return "Error during preprocessing, could not predict."
    
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return CATEGORIES[predicted_class]

def open_webcam():
    """Opens the webcam, detects faces in real-time, and predicts emotion for each face."""
    model = load_model(FINE_TUNED_MODEL_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break
        
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            emotion = predict_emotion(face, model)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def upload_photo():
    """Allows the user to upload a photo, detects the face, and predicts the emotion."""
    file_path = filedialog.askopenfilename()

    if not file_path:
        return
    
    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Error", f"Could not load image at {file_path}")
        return
    
    model = load_model(FINE_TUNED_MODEL_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        messagebox.showinfo("No Faces Detected", "No faces found in the image.")
        return

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        emotion = predict_emotion(face, model)
        cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Emotion Recognition", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_button(master, text, command):
    """Creates a rounded button with hover effect."""
    button_frame = tk.Frame(master, bg="white", bd=1, relief="raised", borderwidth=2)
    button_frame.pack(pady=10)

    button = tk.Button(
        button_frame, 
        text=text, 
        command=command, 
        bg="white", 
        fg="black", 
        font=("Helvetica", 12, "bold"),
        borderwidth=0,
        highlightthickness=0
    )

    button.pack(padx=10, pady=10)

    # Hover effect functions
    def on_enter(event):
        button_frame.config(bg="#eaeaea")  # Change background on hover

    def on_leave(event):
        button_frame.config(bg="white")  # Revert background on leave

    button_frame.bind("<Enter>", on_enter)
    button_frame.bind("<Leave>", on_leave)

    return button

def create_gui():
    """Creates a simple GUI with options to either upload a photo or open the webcam."""
    root = tk.Tk()
    root.title("Facial Emotion Recognition")
    root.geometry("400x300")
    root.configure(bg="white")  # White background

    # Title Label
    title_label = tk.Label(root, text="Facial Emotion Recognition", font=("Helvetica", 16, "bold"), bg="white", fg="black")
    title_label.pack(pady=20)

    # Create buttons with rounded corners and hover effects
    btn_photo = create_button(root, "Upload Photo", upload_photo)
    btn_webcam = create_button(root, "Open Webcam", open_webcam)

    # Run the main loop
    root.mainloop()

if __name__ == "__main__":
    create_gui()
