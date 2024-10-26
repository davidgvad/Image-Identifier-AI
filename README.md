![License](https://img.shields.io/github/license/davidgvad/Tree-Classifier)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)

## Project Structure


```plaintext
Image-Identifier-AI/
│
├── data/
│   ├── raw/            # Original dataset
|
├── results/
│   ├── Result.jpg/
|   ├── result.txt/        # Results
│
├── notebooks/
│   └── Image_Classification.ipynb  # Jupyter Notebook for exploration and analysis
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data loading and preprocessing
│   ├── model_training.py          # Model building and training
│   └── prediction.py              # Making predictions on new data
│
├── requirements.txt    # Python dependencies
├── README.md
├── LICENSE
└── .gitignore
```

## Model Architecture

This document outlines the convolutional neural network (CNN) architecture used in the Image-Identifier-AI project, 
designed for the classification of images into predefined categories based on their visual content. 
The CNN model processes the images through a series of layers, each designed to recognize and abstract different features from the images, ultimately categorizing them with significant accuracy.

The model comprises several key components:

   - **Input Layer**: Accepts images resized to 180x180 pixels with three channels (RGB), normalizing pixel values between 0 and 1.
   - **Convolutional Layers**: Utilize numerous filters to detect various spatial hierarchies of features.
   - **Max Pooling Layers**: Reduce the spatial dimensions of the output from convolutional layers, which decreases the number of parameters and computation in the network.
   - **Flatten Layer**: Transforms the 2D feature maps into a 1D vector to feed into the dense layers, which interpret these features.
   - **Dense Layers**: A series of fully connected layers that use the features to classify the image into one of the categories.
   - **Output Layer**: A dense layer with a softmax activation function that outputs the probability of the image belonging to each class.

In the Summary: The architecture is especially effective for image classification tasks due to its ability to learn hierarchical feature representations,
making it highly capable of recognizing various visual patterns and details within the images.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Image-Identifier-AI.git
   cd Image-Identifier-AI
   ```
2. **Create a virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Configure environment variables:**
      - Create a `.env` file in the root directory.
      - Add necessary variables as shown below:
        ```
        MODEL_PATH=models/latestmodel317/best_model.h5
        DATA_DIR=data/raw/
        ```

## Usage
This section explains how to train the model and make predictions using the project.

## Data Processing
To preprocess the data for the model, execute the data_preprocessing.py command:

  ```bash
    python src/data_preprocessing.py
  ```
## Steps involved:

Data Loading and Preprocessing

## Training Model
To train the convolutional neural network (CNN) model, execute the model_training.py script:

```bash
    python src/model_training.py
```
## Steps Involved:

  - Data Loading and Preprocessing
  - Model Building
  - Model Training
  - Model Saving

## Making Predictions
    ```bash
    python src/prediction.py --model_path models/latestmodel317/best_model.h5 --images_dir data/toCheck --output_file predictions.json
    ```
## Command Line Arguments
   - --model_path: Path to the trained model file
   - --images_dir: Directory containing images you want to classify.
   - --output_file: (Optional) File to save the prediction results in JSON format.
## Steps Involved:
  - Load the Trained Model
  - Process Images
  - Generate Predictions
  - Save Results

   
