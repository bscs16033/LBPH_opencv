import cv2
import os
import numpy as np
import faceRecognition as fr

#test_img=cv2.imread('9.pgm')
#test_img = cv2.imread('person.jpg')
test_img = cv2.imread('Ismail Zam Zam.jpg')
faces_detected, gray_img = fr.faceDetection(test_img)
print("faces_detected: ", faces_detected)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')

name = {1: "Shahrukh Khan", 2: "Tom Cruise", 3: 'Awais Mushtaq', 4: "Ismail"}


for face in faces_detected:
    (x, y, w, h)=face
    roi_gray = gray_img[y:y+w, x:x+h]
    # confidence of 0 means 100% assurity
    label, confidence = face_recognizer.predict(roi_gray)
    print("label: ", label)
    print("confidence: ", confidence)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    # write name on the top left corner of the box
    # the more the conficence value the, the more unlinkely decision our classifier has made
    if confidence > 50:
        predicted_name = "unknown, "
        predicted_name += str(confidence)
    fr.put_text(test_img, predicted_name, x, y)


resized_img=cv2.resize(test_img, (1000, 700))
#resized_img = test_img
cv2.imshow("Image", resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows

