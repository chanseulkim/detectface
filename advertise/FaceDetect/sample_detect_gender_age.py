# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file

import numpy as np
import argparse
import cv2
import os
import cvlib as cv

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_net = cv2.dnn.readNetFromCaffe('data/deploy_age.prototxt', 'data/age_net.caffemodel')
#age_list = ['(0 ~ 9)', '(10 ~ 19)', '(20 ~ 29)', '(30 ~ 49)', '(50 ~ 100)']
age_list = ['(0 ~ 2)', '(4 ~ 6)', '(8 ~ 12)', '(15 ~ 20)', '(25 ~ 32)', '(38 ~ 43)', '(48 ~ 53)', '(60 ~ 100)']

# load model
model = load_model("pre-trained/gender_detection.model")
if model == None:
    # download pre-trained model file (one-time download)
    dwnld_link = "https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model"
    model_path = get_file("gender_detection.model", dwnld_link, cache_subdir="pre-trained", cache_dir=os.getcwd())
    model = load_model(model_path)
    pass
# open webcam
filename = "C:\\Users\\ckstm\\Desktop\\git\\detectface\\advertise_backup\\FaceDetect\\youtube.avi"
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()
    
classes = ['MAN','WOMAN']

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        exit()

    # apply face detection
    face, confidence = cv.detect_face(frame)

    #print(face)
    #print(confidence)

    # loop through detected faces
    for idx, f in enumerate(face):
        faceIndex = idx
        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0, 255, 0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])
        
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # detect age
        blob = cv2.dnn.blobFromImage(face_crop, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        argidx = age_preds[0].argmax()
        age = age_list[argidx]

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        
        # apply gender detection on face
        conf = model.predict(face_crop)[0]
        print(conf)
        print(classes)

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        #label = "face index : {}, {}: {:.2f}%, age:{}".format(faceIndex, label, conf[idx] * 100, age )
        label = "{}: {:.2f}%, age:{}".format(label, conf[idx] * 100, age )
        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # display output
    cv2.imshow("detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()

