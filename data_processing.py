import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define paths and categories
DATA_DIR = 'C:\\Users\\ASUS\\Downloads\\archive\\train'  # Adjust path to your dataset
PREPROCESSED_DIR = '../preprocessed_data'
CATEGORIES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Image size (resize all images to this)
IMG_SIZE = 128  # or 128 if you want smaller images; this can be dynamically changed

# Function to load and preprocess the dataset
def load_data(img_size=IMG_SIZE):
    """
    Loads and preprocesses images from the dataset directory.
    
    Args:
        img_size (int): The size to which images are resized.
    
    Returns:
        Tuple of numpy arrays: (data, labels)
    """
    data = []
    labels = []

    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        print(f"Processing category: {category}, Path: {path}")
        class_num = CATEGORIES.index(category)  # Assign numerical labels

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                # Read the image using OpenCV
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Unable to read image {img_path}. Skipping.")
                    continue

                # Resize the image to IMG_SIZE x IMG_SIZE
                img = cv2.resize(img, (img_size, img_size))

                # Normalize pixel values to [0, 1]
                img = img.astype(np.float32) / 255.0

                # Append data and labels
                data.append(img)
                labels.append(class_num)

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

    return np.array(data), np.array(labels)

def preprocess_and_save(img_size=IMG_SIZE):
    """
    Loads data, preprocesses it, splits into train/validation/test sets, and saves as .npy files.
    """
    # Load and preprocess data
    X, y = load_data(img_size=img_size)
    print(f"Total samples: {X.shape[0]}")

    # One-hot encode labels
    y_encoded = to_categorical(y, num_classes=len(CATEGORIES))
    print(f"Labels one-hot encoded: {y_encoded.shape}")

    # Split data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Further split training data into training and validation sets (90% train, 10% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # Create directory for preprocessed data if it doesn't exist
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    # Save the preprocessed data as .npy files for reuse
    np.save(os.path.join(PREPROCESSED_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PREPROCESSED_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(PREPROCESSED_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PREPROCESSED_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PREPROCESSED_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(PREPROCESSED_DIR, 'y_test.npy'), y_test)

    print("Data preprocessing and saving completed successfully.")

if __name__ == "__main__":
    preprocess_and_save()  # You can modify IMG_SIZE if you want smaller images
