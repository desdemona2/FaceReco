import os
import time
import cv2 as cv

site = os.path.expanduser("~/.local/lib/python3.9/site-packages/")
if not (os.path.isdir("./samples")):
    os.path.mkdir("samples")


classifier = cv.CascadeClassifier(
    f"{site}/cv2/data/haarcascade_frontalface_default.xml"
)


def face_data(classifier):
    capture = cv.VideoCapture(0)
    samples, name = 0, "Subject"
    while True:
        _, frame = capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        values = classifier.detectMultiScale(gray, 1.5, 2)
        if values is ():
            print("Camera is unable to detect any face :( pls show yourself")
            time.sleep(5)
            continue

        x, y, w, h = values[0]
        face = gray[y : y + h, x : x + w]
        samples += 1

        if cv.waitKey(17) == 13 or samples >= 100:
            break


def write_data(face, sample, name):
    fileName = "_".join(name, sample)


face_data(classifier)