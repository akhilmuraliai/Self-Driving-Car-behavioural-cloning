################### NOTEBOOK IMPORTS ###################

import numpy as np
import pandas as pd

import cv2
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

################# IMPORTING DATA AND MODEL #################

label = pd.read_csv('german-traffic-signs/signnames.csv')
classifier = tf.keras.models.load_model('road-sign-classifier.h5')
test_image = 'test-images/6.jpg'

######################## SETTING CAM ########################

frameWidth= 640         
frameHeight = 480

brightness = 180
threshold = 0.75

font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(test_image)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

####################### HELPER FUNCTIONS ######################

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing_image(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img

while True:
    
    ret, frame = cap.read()
    
    img = np.asarray(frame)
    img = cv2.resize(img, (32, 32))
    img = preprocessing_image(img)
    img = img.reshape(1, 32, 32, 1)
    
    cv2.putText(frame, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
    prediction = np.argmax(classifier.predict(img), axis=-1)
    probability = np.max(classifier.predict(img))
    
    if probability > threshold:
        class_name = label.loc[prediction[0]][1]
        
        cv2.putText(frame, str(class_name), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(round(probability * 100, 2) ) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Result", frame)

    if cv2.waitKey(10) and 0xFF == ord('q'):
        break
    