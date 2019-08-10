
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file

import numpy as np
import argparse
import cv2
import os
import cvlib as cv


                     
class Detector(object):
    # download pre-trained model file (one-time download)
    dwnld_link = "https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model"
    model_path = get_file("gender_detection.model", dwnld_link, cache_subdir="pre-trained", cache_dir=os.getcwd())

    # recognization age
    #MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    MODEL_MEAN_VALUES = (103.939, 116.779, 123.680)
    age_net = cv2.dnn.readNetFromCaffe('data/deploy_age.prototxt', 'data/age_net.caffemodel')
    age_list = ['(0 ~ 9)', '(10 ~ 19)', '(20 ~ 29)', '(30 ~ 49)', '(50 ~ 100)']
    #age_list = ['(0 ~ 2)', '(4 ~ 6)', '(8 ~ 12)', '(15 ~ 20)', '(25 ~ 32)', '(38 ~ 43)', '(48 ~ 53)', '(60 ~ 100)']
    model = load_model(model_path)
    webcam = object()
    gender_type = ['MAN','WOMAN']

    @classmethod
    def __init__(cls):
        # open webcam
        cls.webcam = cv2.VideoCapture(0)
        if not cls.webcam.isOpened():
            print("Could not open webcam")
            exit()
        pass
    
    @classmethod
    def Read(cls):
        if cls.webcam.isOpened() :
            status, frame = cls.webcam.read()
            cls.width, cls.height = frame.shape[:2]

            # apply face detection
            face, confidence = cv.detect_face(frame)

            # loop through detected faces
            for idx, f in enumerate(face):
                # get corner points of face rectangle
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]
                # draw rectangle over face
                cv2.rectangle(frame, (startX, startY), (endX,endY), (0, 255, 0), 2)

                # crop the detected face region
                face_crop = np.copy(frame[startY:endY,startX:endX])
            
                if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                    continue
    
                # detect age
                #blob = cv2.dnn.blobFromImage(face_crop, 1, (227, 227), cls.MODEL_MEAN_VALUES, swapRB=False)
                cv2.resize(frame, (227, 227))
                blob = cv2.dnn.blobFromImage(face_crop, 1, (227, 227), cls.MODEL_MEAN_VALUES, swapRB=False)
                cls.age_net.setInput(blob)
                age_preds = cls.age_net.forward()
        
                # preprocessing for gender detection model
                face_crop = cv2.resize(face_crop, (96, 96))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)
                
                # apply gender detection on face
                conf = cls.model.predict(face_crop)[0]
        
                # get label with max accuracy
                idx = np.argmax(conf)

                gender = cls.gender_type[idx]
                argidx = age_preds[0].argmax()
                age = cls.age_list[argidx]

                #label = cls.gender_type[idx]
                #label = "{}: {:.2f}%, age:{}".format(label, conf[idx] * 100, age )
                #Y = startY - 10 if startY - 10 > 10 else startY + 10
                # write label and confidence above face rectangle
                #cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            # display output
            #cv2.imshow("detection", frame)
            return gender, age, frame

            # press "Q" to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # release resources
                cls.webcam.release()
                cv2.destroyAllWindows()
                #break
            pass
        pass

    pass

#detector = Detector()
#detector.Start()
print("import Detector")