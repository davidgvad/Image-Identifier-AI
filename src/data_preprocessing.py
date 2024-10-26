# src/data_preprocessing.py

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

def load_data(data_dir, img_height=180, img_width=180, batch_size=32, validation_split=0.2, seed=123):
    """
    Loads and preprocesses the image data from the specified directory.

    Parameters:
    - data_dir (str): Path to the dataset directory.
    - img_height (int): Height to resize images.
    - img_width (int): Width to resize images.
    - batch_size (int): Number of samples per batch.
    - validation_split (float): Fraction of data to reserve for validation.
    - seed (int): Seed for reproducibility.

    Returns:
    - train_ds (tf.data.Dataset): Training dataset.
    - val_ds (tf.data.Dataset): Validation dataset.
    - class_names (list): List of class names.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    return train_ds, val_ds, class_names

def get_data_augmentation():
    """
    Creates a data augmentation pipeline.

    Returns:
    - data_augmentation (tf.keras.Sequential): Data augmentation layers.
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])
    return data_augmentation

def visualize_data(train_ds, class_names, num_images=9):
    """
    Visualizes a grid of images from the training dataset.

    Parameters:
    - train_ds (tf.data.Dataset): Training dataset.
    - class_names (list): List of class names.
    - num_images (int): Number of images to display.
    """
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(num_images):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Example usage
    data_dir = '../data/raw'  # Adjust the path as needed
    train_ds, val_ds, class_names = load_data(data_dir)
    print(f"Number of classes: {len(class_names)}")
    visualize_data(train_ds, class_names)
