import numpy as np
import cv2

def bgr2gray(image):
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a BGR image with 3 channels.")
    gray = 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]
    return gray.astype(np.uint8)

def gaussian_kernel(size,sigma):
    k = size // 2
    x, y = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def gaussian_blur(img, kernel_size, sigma):

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    kernel = gaussian_kernel(kernel_size, sigma)

    # blurred_img = filter2d(img, kernel) --> Too slow. Taking minutes!!!!
    blurred_img = cv2.filter2D(img, -1, kernel)
    return blurred_img

def filter2d(img, kernel):

    kernel = np.flipud(np.fliplr(kernel))

    img_h, img_w = img.shape
    kernel_h, kernel_w = kernel.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2

    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    filtered_img = np.zeros_like(img, dtype=np.float64)

    for i in range(img_h):
        for j in range(img_w):

            roi = padded_img[i:i + kernel_h, j:j + kernel_w]
            filtered_img[i, j] = np.sum(roi * kernel)

    return np.clip(filtered_img, 0, 255).astype(np.uint8)

def avg_diff(img_cv,img_ut):
    difference = np.abs(img_cv.astype(np.int16) - img_ut.astype(np.int16))
    avg_difference = np.mean(difference)

    print(f"Average pixel difference: {avg_difference}")

def adaptive_threshold(img, max_value,block_size, C):
    if block_size % 2 == 0:
        raise ValueError("Block size must be odd.")

    pad_size = block_size // 2
    padded_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)

    gk = cv2.getGaussianKernel(block_size, -1)
    gk = gk @ gk.T
    local_threshold = cv2.filter2D(padded_img, -1, gk)


    local_threshold -= C

    thresholded_img = np.where(img > local_threshold[pad_size:-pad_size, pad_size:-pad_size], max_value, 0)


    return thresholded_img.astype(np.uint8)

def getMoments(contour):
    moments = cv2.moments(contour)
    HuMoments = cv2.HuMoments(moments)
    return HuMoments

def featuresDistance(f1, f2):
    return np.linalg.norm(np.array(f1) - np.array(f2))

def perspectiveTransform(img, points,size):
    source = np.array(points,dtype="float32")

    dest = np.array([
        [size, size],
        [0, size],
        [0, 0],
        [size, 0]],
        dtype="float32")

    transform = cv2.getPerspectiveTransform(source, dest)
    warped = cv2.warpPerspective(img, transform, (size, size))
    return warped