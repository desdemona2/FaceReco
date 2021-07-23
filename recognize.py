import cv2 as cv
import os

site = os.path.expanduser("~/.local/lib/python3.9/site-packages/")
classifier = cv.CascadeClassifier(
    f"{site}/cv2/data/haarcascade_frontalface_default.xml"
)


def draw(cords, gray, frame, faceClf):
    for x, y, w, h in cords:
        ids, pred = faceClf.predict(gray[y : y + h, x : x + w])
        if pred > 60:
            if ids == 333:
                name = "Subject 1"
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            elif ids == 222:
                name = "Subject 2"
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                name = "Unknown"
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.putText(
                frame,
                str(name),
                (x + 5, y + h - 10),
                cv.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (255, 255, 255),
            )

    return frame


capture = cv.VideoCapture(0)

faceClf = cv.face.LBPHFaceRecognizer_create()
faceClf.read("./faces.xml")

while True:
    _, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cords = classifier.detectMultiScale(gray, 1.1, 3)
    cv.imshow("Frame", draw(cords, gray, frame, faceClf))
    if cv.waitKey(1) == ord("q"):
        break
capture.release()
cv.destroyAllWindows()
