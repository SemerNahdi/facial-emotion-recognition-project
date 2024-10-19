# scripts/evaluate_model.py

import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths (updated to match training script)
MODEL_DIR = 'C:\\Users\\ASUS\\Desktop\\ING3\\ComputerVision\\Facial Emotion Recognition\\models'
FINE_TUNED_MODEL_PATH = os.path.join(MODEL_DIR, 'fine_tuned_model.keras') 
PREPROCESSED_DIR = 'C:\\Users\\ASUS\\Desktop\\ING3\\ComputerVision\\Facial Emotion Recognition\\preprocessed_data'
EVALUATION_REPORT_PATH = os.path.join(MODEL_DIR, 'classification_report.txt')
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, 'confusion_matrix.png')

# Emotion categories (unchanged)
CATEGORIES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def load_test_data():
    """
    Loads test data from .npy files.

    Returns:
        Tuple of numpy arrays: (X_test, y_test)
    """
    X_test = np.load(os.path.join(PREPROCESSED_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(PREPROCESSED_DIR, 'y_test.npy'))
    return X_test, y_test

def evaluate_model():
    """
    Loads the fine-tuned model, evaluates it on the test set, and generates reports.
    """
    # Load the model
    if not os.path.exists(FINE_TUNED_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {FINE_TUNED_MODEL_PATH}")
    model = load_model(FINE_TUNED_MODEL_PATH)
    print("Model loaded successfully.")

    # Load test data
    X_test, y_test = load_test_data()
    print(f"Evaluating on {X_test.shape[0]} test samples.")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification Report
    report = classification_report(y_true, y_pred_classes, target_names=CATEGORIES)
    with open(EVALUATION_REPORT_PATH, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {EVALUATION_REPORT_PATH}")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")

if __name__ == "__main__":
    evaluate_model()
