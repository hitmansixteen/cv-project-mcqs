import cv2
import numpy as np
import utils as ut
from omr import normalize

def normalize(img):
    gray_image_cv = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_image_ut = ut.bgr2gray(img)
    print(gray_image_cv.shape)

def getAnswers(imgLocation):
    original_image = cv2.imread(imgLocation)
    print(original_image.shape)
    normalized_image = normalize(original_image)
    answers = []
    img = None

    return answers, img

getAnswers("img/answered-sheet-photo.jpg")