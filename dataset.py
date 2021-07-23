import os
import cv2 as cv

site = os.path.expanduser("~/.local/lib/python3.9/site-packages/")
classifier = cv.CascadeClassifier(
    f"{site}/cv2/data/haarcascade_frontalface_default.xml"
)


if not (os.path.isdir("./samples")):
    os.mkdir("./samples")
    print("samples directory created")


def face_data(classifier):
    capture = cv.VideoCapture(0)
    samples, id = 0, 222
    while True:
        _, frame = capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        values = classifier.detectMultiScale(gray, 1.1, 3)
        if len(values) == 0:
            continue

        x, y, w, h = values[0]
        face = gray[y : y + h, x : x + w]
        cv.imshow("Face", face)
        samples += 1

        write_data(face, samples, id)

        if cv.waitKey(4) == ord("q") or samples >= 100:
            break


def write_data(face, sample: int, id: int):
    fileName = "./samples/" + "_".join([str(id), str(sample), ".jpg"])
    cv.imwrite(fileName, face)


face_data(classifier)