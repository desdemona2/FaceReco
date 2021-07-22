import os
import time
import cv2 as cv

site = os.path.expanduser("~/.local/lib/python3.9/site-packages/")
classifier = cv.CascadeClassifier(
    f"{site}/cv2/data/haarcascade_frontalface_default.xml"
)


if not (os.path.isdir("./samples")):
    os.mkdir("./samples")


def face_data(classifier):
    capture = cv.VideoCapture(0)
    samples, name = 0, "Subject"
    if len(os.listdir("./samples")) >= 90:
        print(
            """enough samples are already collected. To collect new samples remove old samples
        or increase the value of if check in face_data function"""
        )
    while True:
        _, frame = capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        values = classifier.detectMultiScale(gray, 2, 2)
        if values == ():
            print("""Camera is unable to detect any face :( pls show yourself""")
            time.sleep(2)
            continue

        x, y, w, h = values[0]
        face = gray[y : y + h, x : x + w]
        cv.imshow("View", face)
        samples += 1

        write_data(face, samples, name)

        if cv.waitKey(4) == ord("q") or samples >= 100:
            break


def write_data(face, sample: int, name: str):
    fileName = "./samples/" + "_".join([name, str(sample), ".jpg"])
    cv.imwrite(fileName, face)


face_data(classifier)