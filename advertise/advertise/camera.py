import cv2
import numpy as np
import threading
import time

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


#faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#for (x,y,w,h) in faces:
#    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#    roi_gray = gray[y:y+h, x:x+w]
#    roi_color = img[y:y+h, x:x+w]
#    eyes = eye_cascade.detectMultiScale(roi_gray)
#    for (ex,ey,ew,eh) in eyes:
#        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


class Capture(object):
    capture = object()
    def ReadLoop(self):
        while True:
            ret, frame = self.capture.read()
            if frame != None:
                cv2.imshow("VideoFrame", frame)
                if cv2.waitKey(1) > 0: 
                    break
                pass
            
        pass
    def StartCapture(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        readingFrame = threading.Thread(target=self.ReadLoop, )
        readingFrame.start()
        pass
    
    def Release(self):
        self.capture.release()
        cv2.destroyAllWindows()
        pass
    pass


cap = Capture();

cap.StartCapture();

for x in range(0,10):
    time.sleep(1)
    pass

