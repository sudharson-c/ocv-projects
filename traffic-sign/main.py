import numpy as np
import cv2
from keras.models import load_model

# IMPORT THE TRAINED MODEL
model = load_model("traffic_model.h5")
print("Model loaded successfully.")

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def equalize(img):
    return cv2.equalizeHist(img)
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0  # Normalize to range 0-1
    return img
def getClassName(classNo):
    classes = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
        'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
        'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
        'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons'
    ]
    return classes[classNo] if 0 <= classNo < len(classes) else "Unknown"


# Path to the image to be predicted
image_path = "test/no-entry.jpg"  # Replace with your image path

# READ IMAGE
imgOriginal = cv2.imread(image_path)
if imgOriginal is None:
    print("Error: Image not found.")
    exit()

# PROCESS IMAGE
img = cv2.resize(imgOriginal, (32, 32))
img = preprocessing(img)
img = img.reshape(1, 32, 32, 1)

# PREDICT IMAGE
predictions = model.predict(img)
classIndex = int(np.argmax(predictions))
probabilityValue = np.max(predictions)

# DISPLAY RESULTS
if probabilityValue > 0.75:  # Adjust threshold as needed
    print(f"Predicted Class: {getClassName(classIndex)}")
    print(f"Probability: {round(probabilityValue * 100, 2)}%")
    cv2.putText(imgOriginal, f"CLASS: {getClassName(classIndex)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, f"PROBABILITY: {round(probabilityValue * 100, 2)}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imshow("Result", imgOriginal)
cv2.waitKey(0)
cv2.destroyAllWindows()
