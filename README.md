# OCTID Image Classification

This repository contains code for training and testing an image classification model using the OCTID dataset. The model is implemented in TensorFlow/Keras and can classify images into different categories based on their content.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Making Predictions](#making-predictions)
- [Acknowledgements](#acknowledgements)

## Overview
The goal of this project is to develop an image classification model using the Optical Coherence Tomography (OCT) Images dataset (OCTID). The model is built using TensorFlow/Keras and trained on a dataset of OCT images to classify them into predefined categories.

## Dataset
The OCTID dataset contains a collection of OCT images categorized into different classes. The dataset can be downloaded from the following link:

- [OCTID Dataset](https://example.com/OCTID-dataset)

Make sure to download and extract the dataset to a directory before running the code.

## Setup
To set up the environment and dependencies for this project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/OCTID-image-classification.git
    cd OCTID-image-classification
    ```

2. **Create and activate a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Training the Model
To train the image classification model, run the following script:

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
train_dir = 'path/to/train'
val_dir = 'path/to/validation'

# Image dimensions
img_height = 224
img_width = 224
batch_size = 32

# Data generators
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=20, validation_data=val_generator)

Evaluating the Model
To evaluate the model on the test dataset, use the following script:

# Define the path to the test dataset
test_dir = 'path/to/test'

# Test data generator
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator, verbose=2)
print(f'Test accuracy: {test_accuracy:.4f}')
print(f'Test loss: {test_loss:.4f}')
Making Predictions
To make predictions on individual images, use the following script:
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess an image
img_path = 'path/to/image.jpg'
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the image

# Predict the class
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# Print the prediction
class_labels = list(test_generator.class_indices.keys())
print(f'Predicted class: {class_labels[predicted_class[0]]}')
