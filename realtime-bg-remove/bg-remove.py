import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

# Load the background image
bg_img = cv2.imread('test.png')

cap = cv2.VideoCapture(0)
cap.set(3, 240)
cap.set(4, 144)
segmentor = SelfiSegmentation()

while True:
    success, img = cap.read()
    if not success:
        break

    bg_img_resized = cv2.resize(bg_img, (img.shape[1], img.shape[0]))

    imgOut = segmentor.removeBG(img, bg_img_resized,0.5)

    cv2.imshow('org video', img)
    cv2.imshow('removed bg video', imgOut)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()