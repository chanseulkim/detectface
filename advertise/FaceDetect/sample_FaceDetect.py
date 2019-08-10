import cv2
import numpy as np
import threading
import time

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)


def DetectFace():
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while (True):
        ret, img = capture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
     
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()
    pass

def Gender(args):
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"
    ageNet = cv2.dnn.readNet(genderModel, genderProto)
 
    genderList = ['Male', 'Female']
 
    blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]
    print("Gender Output : {}".format(genderPreds))
    print("Gender : {}".format(gender))
    pass


DetectFace()
exit(0)
