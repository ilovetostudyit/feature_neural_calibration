#!/usr/bin/python3

import cv2 as cv
import os
import yaml
import numpy as np
import glob
import time
import json

from utils import calc_from_cam_to_map_matrix 
from processing import grey_world, Sobel_3_colors, read_calib_params, correct_distortion, preprocess, sift
import utils


#img = cv.imread("../test_images/test_img.png")
img = cv.imread("../test_images/2019-03-02-151138.jpg")
#img = cv.imread("../test_images/Image.png")

MAX_DESC_MSE = 0.007
MIN_QUALITY = 0.0

img = preprocess(img)

kp, des = sift.detectAndCompute(img, None)
kpdes = list(zip(kp, des))
kpdes = list(filter(lambda k: k[0].response >= MIN_QUALITY, kpdes))

kp = list(map(lambda k: k[0], kpdes))

img0 = img.copy()
img_kp0 = img.copy()

RESET_POINTS = True

points_good = []
des_good = []
if RESET_POINTS:
    points_good = []
    def mouse_cb(event, x, y, flags, param):
        global img_ref
        if event == cv.EVENT_LBUTTONDOWN:
            pt = np.float32((x, y))
            nearest = min(kpdes, key=lambda k: np.linalg.norm(np.float32(k[0].pt) - pt))
            if nearest in points_good:
                return
            points_good.append(nearest)
            des_good.append(nearest[1])

            img_kp = img_kp0.copy()
            img_ref = img0.copy()
            for i in range(len(points_good)):
                kpt = points_good[i][0]
                point = tuple(np.int32(kpt.pt))
                cv.circle(img_kp, point, int(kpt.size / 2), (255, 0, 0), 2)
                cv.circle(img_ref, point, int(kpt.size / 2), (255, 0, 0), 2)
                cv.putText(img_kp, str(i), tuple(np.int32(point) + [7, 7]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
                cv.putText(img_ref, str(i), tuple(np.int32(point) + [7, 7]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
            cv.imshow("Img", img_kp)

    cv.drawKeypoints(img_kp0, kp, img_kp0, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow("Img", img_kp0)
    cv.setMouseCallback("Img", mouse_cb)

    while cv.waitKey(100) == -1:
        pass


points_good = list(map(lambda x: {"pt": [int(x[0].pt[0]), int(x[0].pt[1])], "size": int(x[0].size), "desc": list(map(lambda y: int(y), x[1]))}, points_good))

cv.imwrite("reference.png", img_ref)
json.dump(points_good, open("keypoints.json", "wt"))
