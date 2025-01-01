import numpy as np
import os
import pandas as pd
import random
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths and Hyperparameters
DATA_PATH = "traffic_Data/DATA"
LABEL_FILE = "labels.csv"
BATCH_SIZE = 8
EPOCHS = 25
IMG_DIMS = (32, 32, 3)
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.1

# Load and preprocess images
print("Loading data...")
images, class_labels = [], []
classes = os.listdir(DATA_PATH)
print(f"Total Classes Detected: {len(classes)}")

target_size = (32, 32)  # Define the target size for resizing

for class_id in range(len(classes)):
    class_folder = os.path.join(DATA_PATH, str(class_id))
    for img_file in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img,target_size)
        images.append(img)
        class_labels.append(class_id)
    print(f"Loaded Class {class_id + 1}/{len(classes)}")

images = np.array(images)
class_labels = np.array(class_labels)
print(f"Images shape: {images.shape}, Labels shape: {class_labels.shape}")

# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(images, class_labels, test_size=TEST_RATIO)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=VALIDATION_RATIO)

print("Data shapes:")
print(f"Train: {X_train.shape}, {Y_train.shape}")
print(f"Validation: {X_validation.shape}, {Y_validation.shape}")
print(f"Test: {X_test.shape}, {Y_test.shape}")

# Load Labels
data = pd.read_csv(LABEL_FILE)
print(f"Label data shape: {data.shape}")


# Image Preprocessing
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.equalizeHist(img)  # Histogram equalization
    return img / 255.0  # Normalize

# Update the reshaping in the preprocessing
X_train = np.array([preprocess_image(img) for img in X_train]).reshape(-1, 32, 32, 1)  # Resize to (32, 32, 1) for grayscale
X_validation = np.array([preprocess_image(img) for img in X_validation]).reshape(-1, 32, 32, 1)
X_test = np.array([preprocess_image(img) for img in X_test]).reshape(-1, 32, 32, 1)

# Data Augmentation
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)
datagen.fit(X_train)

# One-Hot Encoding Labels
Y_train = to_categorical(Y_train, len(classes))
Y_validation = to_categorical(Y_validation, len(classes))
Y_test = to_categorical(Y_test, len(classes))


def build_model():
    model = Sequential([
        # First Convolutional Block
        Conv2D(60, (5, 5), activation='relu', input_shape=(32, 32, 1)),
        Conv2D(60, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Second Convolutional Block
        Conv2D(30, (3, 3), activation='relu'),
        Conv2D(30, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Dropout(0.5),

        # Flatten the output to feed into Dense layers
        Flatten(),

        # Dense layer
        Dense(500, activation='relu'),
        Dropout(0.5),

        # Output layer
        Dense(len(classes), activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Model Training
model = build_model()
history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_validation, Y_validation),
    shuffle=True
)

# Model Evaluation
test_score, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f'Test Score: {test_score}')
print(f'Test Accuracy: {test_accuracy}')

# Save the trained model
model.save('traffic_model.h5')
print("Model saved as 'traffic_model.h5'")
