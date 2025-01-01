import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load your trained model (replace with your actual model path)
model = load_model('model.h5')

# Class labels
labels = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
    'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
    'Speed limit (120km/h)', 'No passing', 'No passing veh over 3.5 tons', 'Right-of-way at intersection',
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Veh > 3.5 tons prohibited', 'No entry',
    'General caution', 'Dangerous curve left', 'Dangerous curve right', 'Double curve', 'Bumpy road',
    'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians',
    'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
    'End speed + passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only',
    'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
    'End of no passing', 'End no passing veh > 3.5 tons'
]

# Function to predict and annotate image
def predict_and_annotate(image_path):
    # Load the image
    img = image.load_img(image_path)
    img = img.resize((30,30))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    # Predict the class
    prediction = np.argmax(model.predict(images, batch_size=32), axis=-1)

    # Get the label for the predicted class
    label = labels[prediction[0]]

    # Display the image with the prediction label
    plt.imshow(img)
    plt.title(f'Predicted: {label}')
    plt.axis('off')  # Hide the axis
    plt.show()

# Example usage
image_path = './Test/00073.png'  # Replace with the path to your image
predict_and_annotate(image_path)
