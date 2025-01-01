import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint

def load_data_from_csv(csv_path, base_dir, target_size=(32, 32)):
    data = pd.read_csv(csv_path)
    images = []
    labels = []

    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        img_path = os.path.join(base_dir, row['Path'])
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, target_size)  # Resize image
            images.append(img)
            labels.append(row['ClassId'])

    images = np.array(images, dtype=np.float32) / 255.0  # Normalize
    labels = np.array(labels)
    return images, labels

train_csv = "./Train.csv"  # Replace with actual path
test_csv = "./Test.csv"
train_base_dir = "."
test_base_dir = "."

# Load training data
train_images, train_labels = load_data_from_csv(train_csv, train_base_dir)

# Load test data
test_images, test_labels = load_data_from_csv(test_csv, test_base_dir)

# One-hot encode labels
num_classes = len(np.unique(train_labels))
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = (32, 32, 3)
model = create_model(input_shape, num_classes)


# Use the updated file extension `.keras`
checkpoint_callback = ModelCheckpoint(
    filepath='traffic_sign_model.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Include the callback in your model training
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
        checkpoint_callback
    ]
)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

import matplotlib.pyplot as plt

# Make predictions on test data
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Display some test images with predictions
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i])
    plt.title(f"True: {true_classes[i]}, Pred: {predicted_classes[i]}")
    plt.axis('off')
plt.show()

model.save('model.h5')
