# import os
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk

# # Define paths
# MODEL_DIR = 'C:\\Users\\ASUS\\Desktop\\ING3\\ComputerVision\\Facial Emotion Recognition\\models'
# FINE_TUNED_MODEL_PATH = os.path.join(MODEL_DIR, 'fine_tuned_model.keras')
# CATEGORIES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# IMG_SIZE = 128  # Ensure this matches your model's input size

# def load_and_preprocess_image(image):
#     """
#     Preprocesses an image for prediction.
#     Args:
#         image (numpy array): The image loaded or frame captured from the camera.
#     Returns:
#         numpy array: Preprocessed image ready for prediction.
#     """
#     try:
#         # Resize the image to IMG_SIZE x IMG_SIZE
#         img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
#         # Normalize pixel values to [0, 1]
#         img = img.astype(np.float32) / 255.0
        
#         # Expand dimensions to match model's input shape (1, IMG_SIZE, IMG_SIZE, 3)
#         img = np.expand_dims(img, axis=0)
        
#         return img
#     except Exception as e:
#         print(f"Error during image preprocessing: {e}")
#         return None

# def predict_emotion(image, model):
#     """
#     Predicts the emotion from the given image or frame.
#     Args:
#         image (numpy array): The image or frame to process.
#         model (Keras model): Loaded Keras model.
#     Returns:
#         str: Predicted emotion label.
#     """
#     # Preprocess the image
#     preprocessed_image = load_and_preprocess_image(image)
    
#     if preprocessed_image is None:
#         return "Error during preprocessing, could not predict."
    
#     # Make prediction
#     predictions = model.predict(preprocessed_image)
#     predicted_class = np.argmax(predictions, axis=1)[0]
    
#     # Get the corresponding emotion label
#     predicted_emotion = CATEGORIES[predicted_class]
    
#     return predicted_emotion

# def open_webcam():
#     """
#     Opens the webcam, detects faces in real-time, and predicts emotion for each face.
#     """
#     # Load the model
#     model = load_model(FINE_TUNED_MODEL_PATH)
    
#     # Start webcam
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture image.")
#             break
        
#         # Use only the face region for emotion prediction
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#         # If a face is detected, predict emotion
#         for (x, y, w, h) in faces:
#             face = frame[y:y+h, x:x+w]  # Crop the face
#             emotion = predict_emotion(face, model)
            
#             # Display the predicted emotion on the frame
#             cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
#         # Display the resulting frame
#         cv2.imshow('Emotion Recognition', frame)

#         # Break the loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # When everything is done, release the capture and close windows
#     cap.release()
#     cv2.destroyAllWindows()

# def upload_photo():
#     """
#     Allows the user to upload a photo, detects the face, and predicts the emotion.
#     """
#     # Ask the user to select an image file
#     file_path = filedialog.askopenfilename()

#     if not file_path:
#         print("No file selected.")
#         return
    
#     # Load the image
#     img = cv2.imread(file_path)
#     if img is None:
#         print(f"Error: Could not load image at {file_path}")
#         return
    
#     # Load the model
#     model = load_model(FINE_TUNED_MODEL_PATH)

#     # Use only the face region for emotion prediction
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     if len(faces) == 0:
#         print("No faces detected.")
#         return

#     for (x, y, w, h) in faces:
#         face = img[y:y+h, x:x+w]  # Crop the face
#         emotion = predict_emotion(face, model)

#         # Draw the predicted emotion on the image
#         cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

#     # Show the image with emotion labels
#     cv2.imshow("Emotion Recognition", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def create_gui():
#     """
#     Creates a simple GUI with options to either upload a photo or open the webcam.
#     """
#     # Create the main window
#     root = tk.Tk()
#     root.title("Facial Emotion Recognition")

#     # Create and position buttons
#     btn_photo = tk.Button(root, text="Upload Photo", command=upload_photo, width=30, height=2)
#     btn_photo.pack(pady=20)

#     btn_webcam = tk.Button(root, text="Open Webcam", command=open_webcam, width=30, height=2)
#     btn_webcam.pack(pady=20)

#     # Run the main loop
#     root.mainloop()

# if __name__ == "__main__":
#     create_gui()


import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from deepface import DeepFace
import tkinter as tk
from tkinter import filedialog

# Define paths
MODEL_DIR = 'C:\\Users\\ASUS\\Desktop\\ING3\\ComputerVision\\Facial Emotion Recognition\\models'
FINE_TUNED_MODEL_PATH = os.path.join(MODEL_DIR, 'fine_tuned_model.keras')
CATEGORIES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = 128  # Ensure this matches your model's input size

# Load your emotion detection model
emotion_model = load_model(FINE_TUNED_MODEL_PATH)

def load_and_preprocess_image(image):
    """
    Preprocesses an image for emotion prediction.
    Args:
        image (numpy array): The image loaded or frame captured from the camera.
    Returns:
        numpy array: Preprocessed image ready for prediction.
    """
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_emotion(image):
    """
    Predicts the emotion from the given image or frame.
    Args:
        image (numpy array): The image or frame to process.
    Returns:
        str: Predicted emotion label.
    """
    preprocessed_image = load_and_preprocess_image(image)
    predictions = emotion_model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return CATEGORIES[predicted_class]

def open_webcam():
    """
    Opens the webcam, detects faces in real-time, and predicts emotion, gender, and age for each face.
    """
    detection_enabled = True
    show_gender = False
    show_age = False

    # Create a window with a standard size
    cv2.namedWindow('Real-time Emotion, Gender, and Age Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Real-time Emotion, Gender, and Age Detection', 640, 480)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load the pre-trained SSD model for face detection
    modelFile = "C:\\Users\\ASUS\\Desktop\\ING3\\ComputerVision\\Facial Emotion Recognition\\scripts\\res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "C:\\Users\\ASUS\\Desktop\\ING3\\ComputerVision\\Facial Emotion Recognition\\scripts\\deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Display the raw frame before processing
        cv2.imshow('Real-time Emotion, Gender, and Age Detection - Raw Frame', frame)

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Prepare the frame for SSD face detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104.0, 177.0, 123.0], False, False)
        net.setInput(blob)
        detections = net.forward()

        if detection_enabled:
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Confidence threshold for face detection
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Extract the face ROI (Region of Interest)
                    face_roi = frame[startY:endY, startX:endX]
                    emotion = predict_emotion(face_roi)
                    
                    # Perform gender and age analysis
                    try:
                        results = DeepFace.analyze(face_roi, actions=['gender', 'age'], enforce_detection=False)
                        if isinstance(results, list):
                            result = results[0]  # Select the first face if multiple detected
                        else:
                            result = results

                        gender = result['dominant_gender']
                        age = result['age']

                        # Draw rectangle around face
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

                        # Create label
                        label = f"Emotion: {emotion}"
                        if show_gender:
                            label += f", Gender: {gender}"
                        if show_age:
                            label += f", Age: {age}"

                        # Display the predicted emotion, gender, and age based on the flags
                        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    except Exception as e:
                        print(f"Error in DeepFace analysis: {e}")

        # Display the processed frame
        cv2.imshow('Real-time Emotion, Gender, and Age Detection', frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # ESC key
            break
        elif key == ord('s'):  # 's' key will switch on/off detection
            detection_enabled = not detection_enabled
        elif key == ord('g'):  # 'g' key will toggle gender display
            show_gender = not show_gender
        elif key == ord('a'):  # 'a' key will toggle age display
            show_age = not show_age

    cap.release()
    cv2.destroyAllWindows()

def upload_photo():
    """
    Allows the user to upload a photo, detects the face, and predicts the emotion, gender, and age.
    """
    file_path = filedialog.askopenfilename()
    if not file_path:
        print("No file selected.")
        return

    img = cv2.imread(file_path)
    if img is None:
        print(f"Error: Could not load image at {file_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("No faces detected.")
        return

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        emotion = predict_emotion(face)
        
        # Perform gender and age analysis
        try:
            results = DeepFace.analyze(face, actions=['gender', 'age'], enforce_detection=False)
            if isinstance(results, list):
                result = results[0]
            else:
                result = results

            gender = result['dominant_gender']
            age = result['age']

            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # Create label
            label = f"Emotion: {emotion}, Gender: {gender}, Age: {age}"
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        except Exception as e:
            print(f"Error in DeepFace analysis: {e}")

    cv2.imshow("Emotion, Gender, and Age Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_gui():
    """
    Creates a simple GUI with options to either upload a photo or open the webcam.
    """
    root = tk.Tk()
    root.title("Facial Emotion, Gender, and Age Recognition")

    btn_photo = tk.Button(root, text="Upload Photo", command=upload_photo, width=30, height=2)
    btn_photo.pack(pady=20)

    btn_webcam = tk.Button(root, text="Open Webcam", command=open_webcam, width=30, height=2)
    btn_webcam.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
