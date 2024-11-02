# scripts/plot_history.py

import os
import numpy as np
import matplotlib.pyplot as plt

# Define paths
MODEL_DIR = r'C:\Users\ASUS\Desktop\ING3\ComputerVision\Facial Emotion Recognition\models'
INITIAL_HISTORY_PATH = os.path.join(MODEL_DIR, 'initial_training_history.npy')
FINE_TUNE_HISTORY_PATH = os.path.join(MODEL_DIR, 'fine_tune_history.npy')
PLOT_PATH = os.path.join(MODEL_DIR, 'training_plots.png')

def plot_history():
    """
    Plots training and validation accuracy and loss over epochs.
    """
    # Load histories
    initial_history = np.load(INITIAL_HISTORY_PATH, allow_pickle=True).item()
    fine_tune_history = np.load(FINE_TUNE_HISTORY_PATH, allow_pickle=True).item()

    # Combine histories
    acc = initial_history['accuracy'] + fine_tune_history['accuracy']
    val_acc = initial_history['val_accuracy'] + fine_tune_history['val_accuracy']
    loss = initial_history['loss'] + fine_tune_history['loss']
    val_loss = initial_history['val_loss'] + fine_tune_history['val_loss']

    epochs_initial = range(1, len(initial_history['accuracy']) + 1)
    epochs_fine_tune = range(len(initial_history['accuracy']) + 1, len(acc) + 1)
    epochs = list(epochs_initial) + list(epochs_fine_tune)

    # Plot Accuracy
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"Training plots saved to {PLOT_PATH}")

if __name__ == "__main__":
    plot_history()
