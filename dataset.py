import os
import time
import cv2 as cv

site = os.path.expanduser("~/.local/lib/python3.9/site-packages/")


if not (os.path.isdir("./samples")):
    os.mkdir("./samples")


classifier = cv.CascadeClassifier(
    f"{site}/cv2/data/haarcascade_frontalface_default.xml"
)


def face_data(classifier):
    capture = cv.VideoCapture(0)
    samples, name = 0, "Subject"
    while True:
        _, frame = capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        values = classifier.detectMultiScale(gray, 2, 2)
        if values == ():
            print("Camera is unable to detect any face :( pls show yourself")
            time.sleep(5)
            continue

        x, y, w, h = values[0]
        face = gray[y : y + h, x : x + w]
        cv.imshow("View", face)
        samples += 1

        write_data(face, samples, name)

        if cv.waitKey(17) == 5 or samples >= 100:
            break


def write_data(face, sample: int, name: str):
    if len(os.listdir("./samples")) > 80:
        print(
            """Samples are already recorded. To record new samples delete old samples
        or increase the number of samples in face_data function"""
        )
        return None
    fileName = "./samples/" + "_".join([name, str(sample), ".jpg"])
    cv.imwrite(fileName, face)


face_data(classifier)