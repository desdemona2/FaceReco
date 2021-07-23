import cv2 as cv
import numpy as np
import os


def trainer(directory):
    faces = os.listdir(directory)
    # convert id to int before passing to train and convert to numpy array
    ids = np.array([int(face.split("_")[0]) for face in faces])
    frames = []
    for face in faces:
        frame = cv.imread(f"{directory}/{face}")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames.append(gray)

    classifier = cv.face.LBPHFaceRecognizer_create()
    classifier.train(frames, ids)
    classifier.write("./faces.xml")


trainer("./samples")
