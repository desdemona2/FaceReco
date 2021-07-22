# This is real time face detection using Opencv Python

This process works in 3 main steps

## In 1st step we record the sample of the Person we want to recognise

This step is done using [dataset.py](./dataset.py). There are few things which should be considered before running [dataset.py](./dataset.py)

1. Change your site folder where you can find the cascade according to your operating system and your config
2. we can also use IP camera with opencv VideoCapture. Simply pass the url of your IP Camera in place of 0. i.e. cv.VideoCapture('312.53.241.52:8080')
3. You can change the sample size by changing the value of samples in dataset.py
4. If you are recording samples for another person keep in mind to change the name of the person otherwise it would create conflict between both persons
