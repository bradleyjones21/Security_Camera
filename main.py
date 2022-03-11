import numpy as np
import cv2 as cv
from datetime import datetime

cameraNumber = 1
fourcc = 0
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

def main():
    cap = cv.VideoCapture()
    cap.open(cameraNumber + 1 + cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv.CAP_PROP_FPS, 60)
    font = cv.FONT_HERSHEY_PLAIN
    width = int(cap.get(3))
    height = int(cap.get(4))
    while True:
        ret,frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 10)
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        now = datetime.now()
        frame = cv.putText(frame, str(now), (200,height-10),font,2,(255,255,255), 2, cv.LINE_AA)

        cv.imshow('webcam', frame)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    return 0



if __name__ == "__main__":
    main()