import cv2
import os
import numpy as np

def faceDetection(test_img):
    # detection works only on grayscale images
    gray_img=cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Load the cascade
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    #detectMultiScale is used to detect the faces. It takes 3 arguments:
    # arg1: grayscale image
    # arg2: scale factor which specifies how much the image size is reduced with each scale. If the face size is greater than
    # the value defined in the cascade xml file above, you might want to scale it down in order for it to be detected
    # value of 1.2 means reduce the size by 20%
    # arg3: minNeighbors specifies how many neighbors each candidate rectange should have
    # Higher value results in less faces detected, but with higher quality.Since the window is going to slide all over the image, there
    # might be too many faces detected around the same face, each face with coordinates that are only a single or two pixels away
    faces=face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
    # returns coordinates of the faces detected in gray_img
    return faces, gray_img

# to generate labels in our training data
# we are passing it a directory, it will go into each of the subdirectorie and fetch all the images in that subdirectory
# all images will be stored with the label of their subdirectory name e.g. s01
def labels_for_training_data(directory):
    faces = []
    faceID = []

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping System Files")
                continue
            
            # basename is the last folder name in the directory
            # e.g. /home/1/files/folders/folder , here basename is folder
            id=os.path.basename(path)
            # add filename at the end of the path
            img_path=os.path.join(path, filename)
            print("img_path: ", img_path)
            print("id: ", id)
            test_img=cv2.imread(img_path)
            # if no image was read
            if test_img is None:
                print("Image not loaded properly")
                continue

            face_rect, gray_img=faceDetection(test_img)
            if len(face_rect)!=1:
                #since we are only assuming a single person in an image, if there are more than one people, skip this image
                continue    

            (x, y, w, h) = face_rect[0]
            # region of interest in the gray image
            # we are cropping the face
            roi_gray=gray_img[y:y+w, x:x+h]
            faces.append(roi_gray)
            faceID.append(int(id))

    return faces, faceID

def train_classifier(faces, faceID):
    # returns the instance of LBPH recognizer
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    # takes the label as a numpy array
    # labels should be a list of integers
    face_recognizer.train(faces, np.array(faceID))
    # returs the trained classifier
    return face_recognizer

# draw bounding box around the face
# takes image and it's rectangular coordinates
def draw_rect(test_img, faceCoord):
    (x, y, w, h) = faceCoord
    # arg1: image
    # arg2: starting coordinate value of the rectangle
    # arg3: ending coordinate values of the rectangle
    # arg4: color of the rectangular box
    # thickness of the box
    cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), thickness=3)

# put the text on the image
#takes image, text and the x y coordinates where the text should be put
def put_text(test_img, text, x, y):    
    # font family, fontScale, font color, lineType = 3, there are different lines defined
    cv2.putText(test_img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
    


