from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file

import numpy as np
import argparse
import cv2
import os
import cvlib as cv
CURRUNT_PATH = os.path.dirname(os.path.realpath(__file__))
print(CURRUNT_PATH)
DATA_PATH = CURRUNT_PATH + "/data"
test_filename = CURRUNT_PATH + "/youtube.avi"
pre_traind_path = CURRUNT_PATH + "/pre-trained/vgg16_weights.h5"
model_path = CURRUNT_PATH + "/pre-trained/gender_detection.model"

WEBCAM = 0
CAM_NUMBER = WEBCAM


class Detector(object):
    model = load_model(model_path)
    if model == None:
        # download pre-trained model file (one-time download)
        dwnld_link = "https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model"
        print("download model")
        model_path = get_file("gender_detection.model", dwnld_link, cache_subdir="pre-trained", cache_dir=os.getcwd())
        model = load_model(model_path)
        pass

    # recognization age
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    #MODEL_MEAN_VALUES = (103.939, 116.779, 123.68)
    age_net = cv2.dnn.readNetFromCaffe(DATA_PATH + "/deploy_age.prototxt", DATA_PATH + "/age_net.caffemodel")
    #age_list = ['(0 ~ 9)', '(10 ~ 19)', '(20 ~ 29)', '(30 ~ 49)', '(50 ~ 100)']
    age_list = ['(0 ~ 2)', '(4 ~ 6)', '(8 ~ 12)', '(15 ~ 20)', '(25 ~ 32)', '(38 ~ 43)', '(48 ~ 53)', '(60 ~ 100)']
    webcam = object()
    gender_type = ['MAN','WOMAN']

    @classmethod
    def __init__(cls):
        # open webcam
        #cls.webcam = cv2.VideoCapture(CAM_NUMBER)
        #if not cls.webcam.isOpened():
        #    print("Could not open webcam")
        #    exit()
        pass
    @classmethod
    def Detect(cls, frame, isShowRect = False):
        gender = "unknown"
        age = "unknown"
        # apply face detection
        face, confidence = cv.detect_face(frame)

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
            blob = cv2.dnn.blobFromImage(face_crop, 1, (227, 227), cls.MODEL_MEAN_VALUES, swapRB=False)
            cls.age_net.setInput(blob)
            cls.age_preds = cls.age_net.forward()
            argidx = cls.age_preds[0].argmax()
            age = cls.age_list[argidx]

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

            if isShowRect:
                label = gender
                label = "{}: {:.2f}%, age:{}".format(label, conf[idx] * 100, age )
                # write label and confidence above face rectangle
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                pass
            pass
        return gender, age
        pass
    @classmethod
    def Read(cls):
        gender ="Unkown"
        age ="Unkown"
        frame = None
        if cls.webcam.isOpened() :
            status, frame = cls.webcam.read()
            #cls.width, cls.height = frame.shape[:2]
            # apply face detection
            face, confidence = cv.detect_face(frame)

            # loop through detected faces
            for idx, f in enumerate(face):
                face_index = idx
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
                cv2.resize(frame, (224, 224))
                blob = cv2.dnn.blobFromImage(face_crop, 1, (224, 224), cls.MODEL_MEAN_VALUES, swapRB=False)
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

                #gender = cls.gender_type[idx]
                argidx = age_preds[0].argmax()
                age = cls.age_list[argidx]

                label = cls.gender_type[idx]
                label = "{}: {:.2f}%, age:{}".format(label, conf[idx] * 100, age )
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                # write label and confidence above face rectangle
                cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #return face_index, gender, age, frame
            return frame
            # display output
            #cv2.imshow("detection", frame)
            pass
        pass
    def Release(cls):
        cls.webcam.release()
        cv2.destroyAllWindows()
        pass
    pass

#detector = Detector()
#detector.Start()
print("import Detector")