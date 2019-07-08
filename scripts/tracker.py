#!/usr/bin/python3

import cv2 as cv
import os
import yaml
import numpy as np
import glob
import time

from utils import calc_from_cam_to_map_matrix, grey_world, Sobel_3_colors, read_calib_params, correct_distortion
import utils


RESET_POINTS = True

points_good = np.float32(
[[[597, 113]],
 [[603,  71]],
 [[651,  70]],
 [[654, 117]]]
)


#img0 = cv.imread("../test_images/test_img.png")
img0 = cv.imread("../test_images/2019-03-02-151346.jpg")

MAX_DESC_MSE = 0.007
MIN_QUALITY = 0.015


def preprocess(img):
    img = grey_world(img)
#    img = correct_distortion(img)
#    img = cv.blur(img, (5,5))
#    img = Sobel_3_colors(img)
#    img = cv.inRange(img, 70, 255)
#    print(img.dtype)
#    img = img[300:]
#    lines = cv.HoughLines(img, 1, np.pi / 180, 10000)
#    lines_img = np.zeros_like(img)
#    print(lines)
#    for line in lines:
#        x1, y1, x2, y2 = line[0]
#        r, th = line[0]
#        a = np.cos(th)
#        b = np.sin(th)
#        x0 = a * r
#        y0 = b * r
#        x1 = int(x0 + 1000*(-b))
#        y1 = int(y0 + 1000*(a))
#        x2 = int(x0 - 1000*(-b))
#        y2 = int(y0 - 1000*(a))
#        cv.line(lines_img, (x1, y1), (x2, y2), 255, 1)
#    cv.imshow("Lines", lines_img)
#    cv.waitKey(1)
#    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


img0 = preprocess(img0)

#points_good = cv.goodFeaturesToTrack(img0, 100, 0.1, 10)
sift = cv.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(img0, None)
kpdes = list(zip(kp, des))
kpdes = list(filter(lambda k: k[0].response >= MIN_QUALITY, kpdes))

kp = list(map(lambda k: k[0], kpdes))

img01 = cv.drawKeypoints(img0, kp, img0, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv.imshow("Kp", img01)
#cv.waitKey()

RESET_POINTS = True

points_good = []
des_good = []
if RESET_POINTS:
    points_good = []
    def mouse_cb(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            pt = np.float32((x, y))
            nearest = min(kpdes, key=lambda k: np.linalg.norm(np.float32(k[0].pt) - pt))
            points_good.append(nearest)
            des_good.append(nearest[1])

            img = img0.copy()
            for i in range(len(points_good)):
                kpt = points_good[i][0]
                point = tuple(np.int32(kpt.pt))
                cv.circle(img, point, int(kpt.size / 2), (255, 0, 0), 2)
                cv.putText(img, str(i), tuple(np.int32(point) + [7, 7]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
            cv.imshow("Img", img)

    cv.imshow("Img", img0)
    cv.setMouseCallback("Img", mouse_cb)

    while cv.waitKey(100) == -1:
        pass

for file in glob.glob("../test_images/*.jpg") + ["../test_images/test_img.png"]:
    img = cv.imread(file)
    #img = img / (np.random.randint(255, 400) / 255)
    #img = np.uint8(img)
    img = preprocess(img)

    #points, statuses, errs = cv.calcOpticalFlowPyrLK(img0, img, points_good, None)

    #points = cv.goodFeaturesToTrack(img, 100, 0.1, 10)
    kp, des = sift.detectAndCompute(img, None)
    kpdes = list(zip(kp, des))

    indices, diffs = utils.find_in_big(des_good, des)

    kpts = list(map(lambda i, d: kp[i] if d < MAX_DESC_MSE else None, indices, diffs))

    #img = cv.drawKeypoints(img, kpts, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for i in range(len(points_good)):
        kpt = kpts[i]
        if kpt is None:
            continue
        point = tuple(np.int32(kpt.pt))
        cv.circle(img, point, int(kpt.size / 2), (255, 0, 0), 2)
        cv.putText(img, str(i), tuple(np.int32(point) + [7, 7]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))

    
#    for i in range(len(points)):
#        point = points[i]
#        status = 1#statuses[i]
#        err = 0#errs[i]
#        if status == 0 or err > MAX_ERR:
#            continue
#        cv.circle(img, tuple(point[0]), 5, (255, 0, 0), 2)
#        cv.putText(img, str(i), tuple(np.int32(point[0] + [7, 7])), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))


    #Mat = calc_from_cam_to_map_matrix(points, real_points)


    cv.imshow("Img", img)
    if cv.waitKey() == ord('q'):
        break
