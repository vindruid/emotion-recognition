#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import model_from_json
from threading import Timer
import time

#-----------------------------
#opencv initialization

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#-----------------------------
#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open("model/facial_expression_model_structure.json", "r").read())
model.load_weights('model/facial_expression_model_weights.h5') #load weights


# In[ ]:


def get_colors_tuple(emotion):
    if emotion in ['happy','neutral']:
        return (0,255,0)
    if emotion in ['angry','fear','surprise']:
        return (0,0,255)
    if emotion in ['disgust','sad']:
        return (255,0,0)
        

def detect_face_emotion(faces):
    total_face = len(faces)
    color_green = (0,255,0)
    
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    emotions_count = [0,0,0,0,0,0,0]
    for (x,y,w,h) in faces:

        detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)

        img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

        predictions = model.predict(img_pixels) #store probabilities of 7 expressions

        #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        max_index = np.argmax(predictions[0])

        emotion = emotions[max_index]
        emotions_count[max_index] = emotions_count[max_index] + 1

        #write emotion text above rectangle
        colors_tuple = get_colors_tuple(emotion)
        cv2.putText(img, emotion, (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, colors_tuple, 2)
        cv2.line(img,(x-10,y),(x+40,y),colors_tuple,2) 
        cv2.line(img,(x,y-10),(x,y+40),colors_tuple,2) 
        
        #process on detected face end
        #-------------------------
    
    #count faces
    cv2.putText(img, 'total faces: ' + str(total_face) , (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color_green, 2)

    for ix in range(len(emotions)):
        emotion_ = emotions[ix]
        count_ = emotions_count[ix]
        cv2.putText(img, emotion_ + ': '  + str(count_) , (50, 200 + (30 * ix )), cv2.FONT_HERSHEY_SIMPLEX, 1, color_green, 2)
        


# In[ ]:


#-----------------------------

duration = 120 # in seconds
timeout = time.time() + duration

cap = cv2.VideoCapture(1) # 0 for computer camera; 1 for extended camera
while(time.time() < timeout):
    _, img = cap.read()

    img = cv2.flip( img, 1 )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    detect_face_emotion(faces)
    cv2.imshow('img',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
        break

#kill open cv things		
cap.release()
cv2.destroyAllWindows()

