# src/prediction.py

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import argparse
import json

def load_trained_model(model_path):
    """
    Loads the trained CNN model from the specified path.

    Parameters:
    - model_path (str): Path to the saved model.

    Returns:
    - model (tf.keras.Model): Loaded CNN model.
    """
    model = tf.keras.models.load_model(model_path)
    return model

def make_predictions(model, image_path, class_labels, top_k=3):
    """
    Makes predictions on a single image and returns the top K classes.

    Parameters:
    - model (tf.keras.Model): The trained CNN model.
    - image_path (str): Path to the image file.
    - class_labels (list): List of class labels.
    - top_k (int): Number of top predictions to return.

    Returns:
    - top_predictions (list): List of top K predicted class labels.
    """
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    predictions = model.predict(img_array)[0]
    top_indices = predictions.argsort()[-top_k:][::-1]
    top_predictions = [class_labels[i] for i in top_indices]
    return top_predictions

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
    - args (argparse.Namespace): Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Image Identifier AI Prediction Script")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--images_dir', type=str, required=True, help='Directory containing images for prediction.')
    parser.add_argument('--output_file', type=str, default='prediction_results.json', help='File to save prediction results.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Define class labels (e.g., years)
    class_labels = [i for i in range(1907, 1959)]
    for year in [1916, 1917, 1918, 1920]:
        class_labels.remove(year)

    # Initialize result dictionary
    resultDict = {}
    for file in os.listdir(args.images_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            resultDict[file] = []

    # Load the trained model
    model = load_trained_model(args.model_path)

    # Make predictions
    for file in resultDict.keys():
        image_path = os.path.join(args.images_dir, file)
        top_predictions = make_predictions(model, image_path, class_labels, top_k=3)
        resultDict[file] = top_predictions

    # Save results to a JSON file
    with open(args.output_file, 'w') as f:
        json.dump(resultDict, f, indent=4)

    print(f"Predictions saved to {args.output_file}")
