import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.utils.generic_utils import custom_object_scope
from tensorflow.keras.losses import MeanAbsoluteErrorq
# Define the custom loss function
custom_objects = {
    "mae": MeanAbsoluteError()
}
# Load the trained autoencoder model
model_path = "./traffic_model.h5"  # Replace with your model's path
autoencoder = load_model(model_path,custom_objects = custom_objects)


# Function to preprocess an image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Resize the image to match the input shape of the model (32x32)
    image = cv2.resize(image, (32, 32))

    # Normalize the image
    image = image.astype("float32") / 255.0

    # Add batch dimension (1, 32, 32, 3)
    image = np.expand_dims(image, axis=0)
    return image


# Function to predict using the autoencoder model
def predict_image(image_path):
    # Preprocess the input image
    preprocessed_image = preprocess_image(image_path)

    # Predict the reconstruction using the autoencoder
    reconstructed_image = autoencoder.predict(preprocessed_image)

    return reconstructed_image


# Main function
def main():
    image_path = "./test/no-entry.jpg"  # Replace with the path to your input image

    try:
        # Predict the image reconstruction
        reconstructed_image = predict_image(image_path)

        # Remove the batch dimension
        reconstructed_image = np.squeeze(reconstructed_image)

        # Convert back to 0-255 range and to uint8
        reconstructed_image = (reconstructed_image * 255).astype("uint8")

        # Display the original and reconstructed images
        original_image = cv2.imread(image_path)
        original_image = cv2.resize(original_image, (256, 256))
        reconstructed_image = cv2.resize(reconstructed_image, (256, 256))

        cv2.imshow("Original Image", original_image)
        cv2.imshow("Reconstructed Image", reconstructed_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
