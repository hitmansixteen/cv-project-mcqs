import cv2
import numpy as np
import utils as ut
import matplotlib.pyplot as plt


def normalize(img):
    # gray_image_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image_ut = ut.bgr2gray(img)

    # blurred_cv = cv2.GaussianBlur(gray_image_cv, (3, 3), 0)
    blurred_ut = ut.gaussian_blur(gray_image_ut,kernel_size=3,sigma=0.8)

    # adaptive_threshold_cv = cv2.adaptiveThreshold(blurred_ut, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 77, 10)
    adaptive_threshold_ut = ut.adaptive_threshold(blurred_ut, 255, 77, 10)

    return adaptive_threshold_ut

def get_approx_contour(contour, tol=.01):
    epsilon = tol * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def get_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return map(get_approx_contour, contours)

def calculate_contour_features(contour):
    moments = cv2.moments(contour)
    # print(moments)
    hu_moments = cv2.HuMoments(moments)
    # print(hu_moments)
    ut.compare_moments(contour)
    return hu_moments

def calculate_corner_features():
    corner_image = cv2.imread('img/corner.png')
    corner_img_gray = ut.bgr2gray(corner_image)

    contours,hierarchy = cv2.findContours(corner_img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print(contours)
    if len(contours) != 2:
        raise RuntimeError('Did not find the expected contours when looking for the corner')

    corner_contour = next(ct for i, ct in enumerate(contours) if hierarchy[0][i][3] != -1)
    # print(corner_contour)
    return calculate_contour_features(corner_contour)

def features_distance(f1, f2):
    return np.linalg.norm(np.array(f1) - np.array(f2))

def get_corners(contours):
    corner_features = calculate_corner_features()
    return sorted(contours,key=lambda c: features_distance(corner_features,calculate_contour_features(c)))[:4]

def getAnswers(imgLocation):
    original_image = cv2.imread(imgLocation)

    normalized_image = normalize(original_image)

    contours = get_contours(normalized_image)

    corners = get_corners(contours)

    # print(corners)

    answers = []
    img = None

    return answers, img

getAnswers("img/answered-sheet-photo.jpg")