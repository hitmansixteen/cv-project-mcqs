import cv2
import numpy as np
import utils as ut

TRANSFORM_SIZE = 512
NO_OF_QUESTIONS = 10
ANSWER_SHEET_WIDTH = 740
ANSWER_SHEET_HEIGHT = 1049
ANSWER_HEIGHT = 50
ANSWER_HEIGHT_WITH_MARGIN = 80
ANSWER_LEFT_MARGIN = 200
ANSWER_RIGHT_MARGIN = 90
FIRST_ANSWER_TOP_Y = 200
OPTION_HEIGHT = 50
OPTION_WIDTH = 50
OPTION_WIDTH_WITH_MARGIN = 100




def normalize(img):
    gray_image_ut = ut.bgr2gray(img)

    blurred_ut = ut.gaussian_blur(gray_image_ut,kernel_size=3,sigma=0.8)

    adaptive_threshold_ut = ut.adaptive_threshold(blurred_ut, 255, 77, 10)

    return adaptive_threshold_ut

def getContours(image):
    contours, _ = cv2.findContours(image,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def getApproxContours(contour, tol=.01):
        epsilon = tol * cv2.arcLength(contour, True)
        approx_contours = cv2.approxPolyDP(contour, epsilon, True)
        return approx_contours

    approx_contours = map(getApproxContours, contours)

    return approx_contours

def calculateCornerMoments():
    corner_img = cv2.imread('img/corner.png')
    corner_img_gray = cv2.cvtColor(corner_img, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(corner_img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    corner_contour = next(ct for i, ct in enumerate(contours) if hierarchy[0][i][3] != -1)

    return ut.getMoments(corner_contour)

def getCorners(contours):
    corner_moments = calculateCornerMoments()
    return sorted(contours,key=lambda c: ut.featuresDistance(corner_moments,ut.getMoments(c)))[:4]

def getOutmost(corners):
    all_points = np.concatenate(corners)
    rect = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rect)
    box_points = np.int32(box)

    origin = np.mean(box_points,axis=0)

    def getAngle(point):
        x,y = point - origin
        angle = np.arctan2(y,x)
        return 2*np.pi + angle if angle < 0 else angle
    
    sorted_points = sorted(box_points, key=getAngle)

    return sorted_points

def sheetCoordinateToTransformedCoordinate(x,y):
    return list(map(lambda n: int(np.round(n)), (TRANSFORM_SIZE * x / ANSWER_SHEET_WIDTH,TRANSFORM_SIZE * y / ANSWER_SHEET_HEIGHT)))

def getQuestion(image, question_no):
    topLeft = sheetCoordinateToTransformedCoordinate(
        ANSWER_LEFT_MARGIN,
        FIRST_ANSWER_TOP_Y + ANSWER_HEIGHT_WITH_MARGIN * question_no
    )

    bottomRight = sheetCoordinateToTransformedCoordinate(
       ANSWER_SHEET_WIDTH - ANSWER_RIGHT_MARGIN,
        FIRST_ANSWER_TOP_Y + ANSWER_HEIGHT + ANSWER_HEIGHT_WITH_MARGIN * question_no
    )

    return image[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]

def getQuestions(image):
    for i in range(NO_OF_QUESTIONS):
        yield getQuestion(image,i)

def getOptions(question):
    for i in range(5):
        x0, _ = sheetCoordinateToTransformedCoordinate(OPTION_WIDTH_WITH_MARGIN * i, 0)
        x1, _ = sheetCoordinateToTransformedCoordinate(OPTION_WIDTH + OPTION_WIDTH_WITH_MARGIN * i, 0)
        yield question[:, x0:x1]

def getMarked(Options):
    means = list(map(np.mean, Options))
    less_than_120 = [mean for mean in means if mean < 120]
    # print(less_than_120)

    if len(less_than_120) == 0:
        return None
    if len(less_than_120) > 1:
        return "N"
    return means.index(less_than_120[0])
    
def getLetter(marked):
    if marked == "N":
        return "N"
    return ["A", "B", "C", "D", "E"][marked] if marked is not None else "N/A"


def getAnswers(imgLocation):
    original_image = cv2.imread(imgLocation)

    normalized_image = normalize(original_image)

    contours = getContours(normalized_image)

    corners = getCorners(contours)

    outmost = getOutmost(corners)

    normalized_transform = ut.perspectiveTransform(normalized_image, outmost,TRANSFORM_SIZE)

    answers = []
    for i, question in enumerate(getQuestions(normalized_transform)):
        # print(f"Question {i+1}:")
        marked = getMarked(getOptions(question))

        answers.append(getLetter(marked))
        

    print(answers)

    return answers
