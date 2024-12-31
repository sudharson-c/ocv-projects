import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameters
path = "myData"  # Folder with all class folders
labelFile = 'labels.csv'  # CSV file with class labels
batch_size_val = 50
epochs_val = 10
image_dimensions = (32, 32, 3)
test_ratio = 0.2
validation_ratio = 0.2

# Loading Images
print("Loading Images...")
images, class_no = [], []
class_folders = os.listdir(path)
print(f"Total Classes Detected: {len(class_folders)}")
for class_id, folder in enumerate(class_folders):
    folder_path = os.path.join(path, folder)
    if not os.path.isdir(folder_path):
        continue
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (image_dimensions[0], image_dimensions[1]))
        images.append(img)
        class_no.append(class_id)
images = np.array(images)
class_no = np.array(class_no)

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(images, class_no, test_size=test_ratio, stratify=class_no)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_ratio, stratify=y_train)

# Data Shapes
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_validation.shape}")
print(f"Test data shape: {X_test.shape}")

# Preprocessing Function
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

# Preprocessing Images
X_train = np.array([preprocess(img) for img in X_train]).reshape(-1, image_dimensions[0], image_dimensions[1], 1)
X_validation = np.array([preprocess(img) for img in X_validation]).reshape(-1, image_dimensions[0], image_dimensions[1], 1)
X_test = np.array([preprocess(img) for img in X_test]).reshape(-1, image_dimensions[0], image_dimensions[1], 1)

# One-hot Encoding Labels
num_classes = len(np.unique(class_no))
y_train = to_categorical(y_train, num_classes)
y_validation = to_categorical(y_validation, num_classes)
y_test = to_categorical(y_test, num_classes)

# Data Augmentation
data_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2,
                               shear_range=0.1, rotation_range=10)
data_gen.fit(X_train)

# Model Definition
def create_model():
    model = Sequential([
        Input(shape=(image_dimensions[0], image_dimensions[1], 1)),
        Conv2D(64, (5, 5), activation='relu'),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Creating and Training the Model
model = create_model()
print(model.summary())
history = model.fit(data_gen.flow(X_train, y_train, batch_size=batch_size_val),
                    validation_data=(X_validation, y_validation),
                    epochs=epochs_val, steps_per_epoch=len(X_train) // batch_size_val, shuffle=True)

# Plotting Training History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.show()

# Evaluating the Model
test_score = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_score[0]}")
print(f"Test Accuracy: {test_score[1]}")

# Saving the Model
model.save("traffic_sign_model.h5")
print("Model saved as traffic_sign_model.h5")
