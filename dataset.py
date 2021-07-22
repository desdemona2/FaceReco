import os
import cv2 as cv

site = os.path.expanduser("~/.local/lib/python3.9/site-packages/")


def dataset():
    classifier = cv.CascadeClassifier(
        f"{site}/cv2/data/haarcascade_frontalface_default.xml"
    )
