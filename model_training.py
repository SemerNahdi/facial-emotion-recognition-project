# scripts/model_training.py

import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard

# Define paths
PREPROCESSED_DIR = 'C:\\Users\\ASUS\\Desktop\\ING3\\ComputerVision\\Facial Emotion Recognition\\preprocessed_data'
MODEL_DIR = 'C:\\Users\\ASUS\\Desktop\\ING3\\ComputerVision\\Facial Emotion Recognition\\models'
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')
LOG_DIR = os.path.join(MODEL_DIR, 'logs')

# Create necessary directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Model file paths
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.keras')
FINE_TUNED_MODEL_PATH = os.path.join(MODEL_DIR, 'fine_tuned_model.keras')
TRAIN_HISTORY_CSV = os.path.join(MODEL_DIR, 'training_history.csv')
FINE_TUNE_HISTORY_CSV = os.path.join(MODEL_DIR, 'fine_tune_training_history.csv')

# Load preprocessed data
def load_preprocessed_data():
    """
    Loads preprocessed data from .npy files.

    Returns:
        Tuple of numpy arrays: (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    X_train = np.load(os.path.join(PREPROCESSED_DIR, 'X_train.npy'))
    X_val = np.load(os.path.join(PREPROCESSED_DIR, 'X_val.npy'))
    X_test = np.load(os.path.join(PREPROCESSED_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(PREPROCESSED_DIR, 'y_train.npy'))
    y_val = np.load(os.path.join(PREPROCESSED_DIR, 'y_val.npy'))
    y_test = np.load(os.path.join(PREPROCESSED_DIR, 'y_test.npy'))

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def build_model(num_classes):
    """
    Builds and compiles the emotion recognition model using VGG16 as the base.

    Args:
        num_classes (int): Number of emotion categories.

    Returns:
        Compiled Keras model.
    """
    # Load the VGG16 model without the top classification layers
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))


    # Freeze all layers in the base model initially
    vgg_base.trainable = False

    # Create a new model on top
    model = Sequential([
        vgg_base,
        GlobalAveragePooling2D(),  # Efficiently reduces feature maps
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Output layer
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def fine_tune_model(model, base_model, train_data, val_data, epochs=20, batch_size=32):
    """
    Fine-tunes the pre-trained base model by unfreezing some layers.

    Args:
        model (Keras model): The compiled model.
        base_model (Keras model): The pre-trained base model.
        train_data (tuple): Tuple of (X_train, y_train).
        val_data (tuple): Tuple of (X_val, y_val).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        Trained Keras model and its history.
    """
    # Unfreeze the top layers of the base model for fine-tuning
    # For VGG16, we'll unfreeze the last 4 convolutional layers
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    # Recompile the model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Define callbacks for fine-tuning
    fine_tune_checkpoint = ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, 'fine_tuned_best_model.keras'),  # Change here
        monitor='val_loss',
        save_best_only=True,
        verbose=1
        )
    fine_tune_early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    fine_tune_csv_logger = CSVLogger(
        FINE_TUNE_HISTORY_CSV,
        append=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        verbose=1,
        min_lr=1e-7
    )
    tensorboard = TensorBoard(
        log_dir=os.path.join(LOG_DIR, 'fine_tune_logs'),
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )

    # Train the model
    history = model.fit(
        train_data[0], train_data[1],
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[fine_tune_checkpoint, fine_tune_early_stop, fine_tune_csv_logger, reduce_lr, tensorboard]
    )

    return model, history

def main():
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_preprocessed_data()
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")

    # Get number of classes from labels
    num_classes = y_train.shape[1]

    # Check if a best model already exists to resume training
    if os.path.exists(BEST_MODEL_PATH):
        print("Loading existing best model to resume training...")
        model = load_model(BEST_MODEL_PATH)
    else:
        # Build the model
        model = build_model(num_classes)
        print("Initial Model Summary:")
        model.summary()

    # Define callbacks for initial training
    checkpoint = ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    csv_logger = CSVLogger(
        TRAIN_HISTORY_CSV,
        append=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        verbose=1,
        min_lr=1e-7
    )
    tensorboard = TensorBoard(
        log_dir=os.path.join(LOG_DIR, 'initial_training_logs'),
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )

    # Train the model initially (with base model frozen)
    initial_history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        callbacks=[checkpoint, early_stop, csv_logger, reduce_lr, tensorboard]
    )

    # Save initial training history
    np.save(os.path.join(MODEL_DIR, 'initial_training_history.npy'), initial_history.history)
    print("Initial training history saved.")

    # Load the base model for fine-tuning
    base_model = model.layers[0]

    # Fine-tune the model by unfreezing some layers
    fine_tuned_model, fine_tune_history = fine_tune_model(
        model, base_model,
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=20,
        batch_size=32
    )

    # Save the final fine-tuned model
    fine_tuned_model.save(FINE_TUNED_MODEL_PATH)
    print(f"Fine-tuned model saved to {FINE_TUNED_MODEL_PATH}")

    # Evaluate the model on the test set
    test_loss, test_accuracy = fine_tuned_model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Save fine-tuning history
    np.save(os.path.join(MODEL_DIR, 'fine_tune_history.npy'), fine_tune_history.history)
    print("Fine-tuning history saved.")

if __name__ == "__main__":
    main()
