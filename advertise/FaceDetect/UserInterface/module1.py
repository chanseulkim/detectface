
import sys
import Detector
import cv2
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton

detector = Detector.Detector()
camera = cv2.VideoCapture(0)

class ShowVideo(QtCore.QObject):
    isDetecting = False
    isShow = 1
    run_video = True
    #image = detector.Read()
    status, image = camera.read()
    height, width = image.shape[:2]

    VideoSignal1 = QtCore.pyqtSignal(QtGui.QImage)
    VideoSignal2 = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(ShowVideo, self).__init__(parent)

    #영상 감지를 할지 안할지는 Notify를 받고 결정됨
    @QtCore.pyqtSlot()
    def StartVideo(self):
        global image
        while self.run_video:
            status, image = camera.read()
            if status :
                if (self.isDetecting):
                    gender, age = detector.Detect(image, True)
                    print("{}, {} ".format(gender, age))
                self.ShowFrame(image)        
            
            pass
        camera.release()

    @QtCore.pyqtSlot()
    def StopVideo(self):
        self.run_video = False
        pass

    #영상 감시 시작을 알림
    @QtCore.pyqtSlot()
    def NotifyDetect(self):
        self.isDetecting = True

    @QtCore.pyqtSlot()
    def ShowFrame(self, image):
        color_swapped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        qt_image1 = QtGui.QImage(color_swapped_image.data, self.width, self.height, color_swapped_image.strides[0], QtGui.QImage.Format_RGB888)
        self.VideoSignal1.emit(qt_image1)

        #감지중일때만 추천 광고영상을 재생시켜줌
        if self.isDetecting:
            ad_video = cv2.VideoCapture(Detector.test_filename)
            s, img = ad_video.read()
            #img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #img_canny = cv2.Canny(img_gray, 50, 100)
            #qt_image2 = QtGui.QImage(img_canny.data, self.width, self.height, img_canny.strides[0], QtGui.QImage.Format_Grayscale8)
            self.VideoSignal2.emit(img)
            pass
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(25, loop.quit) #25 ms
        loop.exec_()
        pass

class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

    def initUI(self):
        self.setWindowTitle('Test')

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    thread = QtCore.QThread()
    thread.start()
    vid = ShowVideo()
    vid.moveToThread(thread)
    webcam_viewer = ImageViewer()
    advertisment_viewer = ImageViewer()

    vid.VideoSignal1.connect(webcam_viewer.setImage)
    vid.VideoSignal2.connect(advertisment_viewer.setImage)
    start_button1 = QtWidgets.QPushButton('Start detect')
    start_button1.clicked.connect(vid.NotifyDetect)
    start_button1.move(80,13)
    
    stop_button1 = QtWidgets.QPushButton('Stop detect')
    stop_button1.clicked.connect(vid.StopVideo)
    stop_button1.move(80,13)

    vertical_layout = QtWidgets.QVBoxLayout()
    horizontal_layout = QtWidgets.QHBoxLayout()
    horizontal_layout.addWidget(webcam_viewer)
    horizontal_layout.addWidget(advertisment_viewer)
    vertical_layout.addLayout(horizontal_layout)
    vertical_layout.addWidget(start_button1)
    vertical_layout.addWidget(stop_button1)

    layout_widget = QtWidgets.QWidget()
    layout_widget.setLayout(vertical_layout)

    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.show()
    vid.StartVideo()
    sys.exit(app.exec_())