import cv2
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from django.conf import settings
import subprocess
import random


def binarize(image):
    img = np.array(Image.open(BytesIO(image)))
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    ret, format = cv2.imencode(".png", thresh)
    buf = BytesIO(format.tobytes())
    image = 'data:image/png;base64,' + base64.b64encode(buf.read()).decode("ascii")
    return image


def face_detect(image):
    img = np.array(Image.open(BytesIO(image)))
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(settings.BASE_DIR + "/data/haarcascade_frontalface_alt2.xml")
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30))

    if len(rects) != 0:
        rects[:, 2:] += rects[:, :2]
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ret, format = cv2.imencode(".png", image_rgb)
    buf = BytesIO(format.tobytes())
    image = 'data:image/png;base64,' + base64.b64encode(buf.read()).decode("ascii")

    return image


def text_detect(image):
    filename = "/tmp/" + str(random.randint(100000000, 1000000000))
    with open(filename, 'wb') as file:
        file.write(BytesIO(image).read())
    p = subprocess.getoutput("/home/murat/text " + filename)
    file = open(p, "rb")
    image = 'data:image/png;base64,' + base64.b64encode(file.read()).decode("ascii")

    return image