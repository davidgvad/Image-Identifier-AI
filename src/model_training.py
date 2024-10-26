# src/model_training.py

import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_preprocessing import load_data, get_data_augmentation
import os

def build_model(input_shape, num_classes, data_augmentation):
    """
    Builds the CNN model architecture.

    Parameters:
    - input_shape (tuple): Shape of the input images.
    - num_classes (int): Number of output classes.
    - data_augmentation (tf.keras.Sequential): Data augmentation layers.

    Returns:
    - model (tf.keras.Model): Uncompiled CNN model.
    """
    model = tf.keras.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),
        data_augmentation,
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def compile_model(model):
    """
    Compiles the CNN model with optimizer, loss function, and metrics.

    Parameters:
    - model (tf.keras.Model): The CNN model to compile.

    Returns:
    - model (tf.keras.Model): Compiled CNN model.
    """
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, train_ds, val_ds, epochs=30, model_dir='models/latestmodel317'):
    """
    Trains the CNN model and saves the best model.

    Parameters:
    - model (tf.keras.Model): The compiled CNN model.
    - train_ds (tf.data.Dataset): Training dataset.
    - val_ds (tf.data.Dataset): Validation dataset.
    - epochs (int): Number of training epochs.
    - model_dir (str): Directory to save the trained model.

    Returns:
    - history (tf.keras.callbacks.History): Training history.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Define callbacks for saving the best model and early stopping
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_dir, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint, early_stop]
    )
    return history

if __name__ == "__main__":
    # Example usage
    data_dir = '../data/raw'  # Adjust the path as needed
    train_ds, val_ds, class_names = load_data(data_dir)
    data_augmentation = get_data_augmentation()
    input_shape = (180, 180, 3)
    num_classes = len(class_names)

    model = build_model(input_shape, num_classes, data_augmentation)
    model = compile_model(model)
    model.summary()

    history = train_model(model, train_ds, val_ds)
