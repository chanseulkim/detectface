import os
import datetime
import cv2
import sys


def MakeDir(path):
    try:
        if not (os.path.isdir(Recorder.rec_dir)):
            os.makedirs(os.path.join(Recorder.rec_dir))
    except OSError as e :
        if e.errno != errno.EEXIST:
            print("Failed to create directory")
        raise
class Recorder(object):
    rec_dir = ".\\rec\\"
    record_path = rec_dir + "rec"
    record_ext = ".avi"
    is_recording = False
    capture = object()
    capture_ext = ".png"
    def OpenCapture(self, args):
        MediaClient.rtsp_uri = args
        self.capture = cv2.VideoCapture(MediaClient.rtsp_uri)
        self.capture.open(self.GetImageFilename())
        return self.capture
        pass
    def GetRecFilename():
        now = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
        return self.record_path + now + self.record_ext
        pass
    def GetImageFilename(self):
        now = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
        return self.rec_dir + "capture_image" + now + self.capture_ext
        pass
    pass
class MediaClient(object):
    rtsp_uri = "rtsp://root:pass@172.16.10.109/AVStream1_1" # default
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    pass
class UiTools(object):
    
    def GetUri():
        print("- default uri: rtsp://root:pass@172.16.10.109/AVStream1_1")
        print("- Input rtsp uri : ")
        readline_value = sys.stdin.readline()
        print("- (1) Capture image = Ctrl + Z \n- (2) Start record = Ctrl + X \n- (3) Stop record = Ctrl + C")
        if readline_value != '\n':
            readline_value = readline_value.rstrip('\n')
            return readline_value;
        return "rtsp://root:pass@172.16.10.109/AVStream1_1"
        pass
    pass
def Main(args):
    rtsp_uri = args
    recorder = Recorder()
    MakeDir(recorder.rec_dir)
    rtsp_uri = UiTools.GetUri()
    while True:
        capture = recorder.OpenCapture(rtsp_uri)
        if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
            ret, frame = capture.read()
            cv2.imshow("VideoFrame", frame) #now = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
            key = cv2.waitKey(33)
            if key == 27:
                break
            elif key == 26: # Ctrl + Z
                print("Image captured")
                cv2.imwrite(recorder.GetImageFilename(), frame)
            elif key == 24 : # Ctrl + X
                print("Recording Started")
                record = True
                video = cv2.VideoWriter(recorder.GetRecFilename(), fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            elif key == 3: # Ctrl + C
                print("Recording Stoped")
                record = False
                if "video" in locals():
                    video.release()
                    pass
            
            if Recorder.is_recording == True:
                video.write(frame)
                pass
        capture.release()
        cv2.destroyAllWindows()
    pass
Main("rtsp://root:pass@172.16.10.109/AVStream1_1")