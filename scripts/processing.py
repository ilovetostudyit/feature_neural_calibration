import numpy as np
import os
import yaml
import cv2 as cv


#sift = cv.xfeatures2d.SIFT_create(contrastThreshold=0.007, edgeThreshold=2)
#sift = cv.xfeatures2d.SURF_create()
sift = cv.xfeatures2d.SIFT_create()



def preprocess(img):
    img = grey_world(img)
    img = correct_distortion(img)
    img = cv.GaussianBlur(img, (11, 11), 0)
    return img



def grey_world(image):
    mu = np.average(image, axis=(0, 1))
    image = image * mu[1] / mu
    image = np.clip(image, 0, 255)
    return np.uint8(image)


def read_calib_params():
    global camera_matrix
    global distCoeffs
    global newmat

    yamlstr = open(os.path.join(os.path.dirname(__file__), "../params/camera_calibration.yml"), "rt").read()
    calibr_params = yaml.load(yamlstr)
    camera_matrix = calibr_params["camera_matrix"]["data"]
    camera_matrix = np.float32(camera_matrix)
    camera_matrix = np.reshape(camera_matrix, (3,3))

    distCoeffs = calibr_params["distortion_coefficients"]["data"]
    distCoeffs = np.float32(distCoeffs)
    distCoeffs = np.reshape(distCoeffs, (5,))

    h, w = 960, 1280
    newmat, roi = cv.getOptimalNewCameraMatrix(camera_matrix,distCoeffs,(w,h),0.3,(w,h),1)

def correct_distortion(img):
    if not 'camera_matrix' in globals():
        read_calib_params()
    return cv.undistort(img, camera_matrix, distCoeffs, None, newmat)

def Sobel_1_color(gray):
    ddepth = cv.CV_16S
    mix = 0.5

    grad_x = cv.Sobel(gray, ddepth = ddepth, dx = 1, dy = 0)
    grad_y = cv.Sobel(gray, ddepth = ddepth, dx = 0, dy = 1)
    grad_x_abs = cv.convertScaleAbs(grad_x)
    grad_y_abs = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(grad_x_abs, mix, grad_y_abs, mix, 0)

    return grad

def Sobel_3_colors(img):
    b,g,r = cv.split(img)
    mix = 0.7

    grad_r = Sobel_1_color(r)
    grad_g = Sobel_1_color(g)
    grad_b = Sobel_1_color(b)

    grad = cv.addWeighted(grad_r, mix, grad_g, mix, 0)
    grad = cv.addWeighted(grad, 2 * mix, grad_b, mix, 0)

    return grad



