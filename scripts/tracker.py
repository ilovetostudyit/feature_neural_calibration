#!/usr/bin/python3

import cv2 as cv
import os
import yaml
import numpy as np
import glob
import time

from utils import calc_from_cam_to_map_matrix


MAX_ERR = 8
RESET_POINTS = True

points_good = np.float32(
[[[597, 113]],
 [[603,  71]],
 [[651,  70]],
 [[654, 117]]]
)


#img0 = cv.imread("../test_images/test_img.png")
img0 = cv.imread("../test_images/2019-03-02-150600.jpg")


def grey_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/np.average(nimg[2])),255)
    return  nimg.transpose(1, 2, 0).astype(np.uint8)

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

def preprocess(img):
    img = grey_world(img)
    img = correct_distortion(img)
    #img = cv.blur(img, (5,5))
    #img = Sobel_3_colors(img)
    img = img[300:]
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


img0 = preprocess(img0)

RESET_POINTS = False

if RESET_POINTS:
    points_good = []
    def mouse_cb(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            points_good.append([[x, y]])

        img = img0.copy()
        for i in range(len(points_good)):
            point = points_good[i]
            cv.circle(img, tuple(point[0]), 5, (255, 0, 0), 2)
            cv.putText(img, str(i), tuple(np.int32(point[0]) + [7, 7]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
        cv.imshow("Img", img)

    cv.imshow("Img", img0)
    cv.setMouseCallback("Img", mouse_cb)

    while cv.waitKey(100) == -1:
        pass
    points_good = np.float32(points_good)
    print(points_good)

for file in glob.glob("../test_images/*.jpg"):
    img = cv.imread(file)
    #img = img / (np.random.randint(255, 400) / 255)
    #img = np.uint8(img)
    img = preprocess(img)

    points, statuses, errs = cv.calcOpticalFlowPyrLK(img0, img, points_good, None)
    cv.calcOpticalFlowPyrLK

    points = cv.goodFeaturesToTrack(img, 100, 0.1, 10)

    for i in range(len(points)):
        point = points[i]
        status = 1#statuses[i]
        err = 0#errs[i]
        if status == 0 or err > MAX_ERR:
            continue
        cv.circle(img, tuple(point[0]), 5, (255, 0, 0), 2)
        cv.putText(img, str(i), tuple(np.int32(point[0] + [7, 7])), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))


    #Mat = calc_from_cam_to_map_matrix(points, real_points)


    cv.imshow("Img", img)
    if cv.waitKey() == ord('q'):
        break
