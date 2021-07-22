import os
import cv2 as cv

site = os.path.expanduser("~/.local/lib/python3.9/site-packages/")


classifier = cv.CascadeClassifier(
    f"{site}/cv2/data/haarcascade_frontalface_default.xml"
)


def face_data(classifier):
    capture = cv.VideoCapture(0)
    _, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


def write_data(face):
    pass
