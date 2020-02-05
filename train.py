import cv2
import os
import numpy as np
import faceRecognition as fr

faces, faceID = fr.labels_for_training_data('Dataset/train')
face_recognizer = fr.train_classifier(faces, faceID)
face_recognizer.save("trainingData.yml")

