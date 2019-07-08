#!/usr/bin/python3

import cv2 as cv
import os
import yaml
import numpy as np
import glob
import time
import json

from utils import calc_from_cam_to_map_matrix, grey_world, Sobel_3_colors, read_calib_params, correct_distortion
import utils
import coords


MAX_DESC_MSE = 0.007
MIN_QUALITY = 0.0

with open("keypoints.json", "rt") as kpfile:
    points_good = json.load(kpfile)
des_good = list(map(lambda x: x["desc"], points_good))

def preprocess(img):
    img = grey_world(img)
    img = correct_distortion(img)
    return img

sift = cv.xfeatures2d.SIFT_create(contrastThreshold=0.007, edgeThreshold=2)

for file in glob.glob("../test_images/*.jpg") + ["../test_images/test_img.png"] + ["../test_images/Image.png"]:
    img = cv.imread(file)
    img = preprocess(img)

    kp, des = sift.detectAndCompute(img, None)
    kpdes = list(zip(kp, des))

    indices, diffs = utils.find_in_big(des_good, des)

    kpts = []#list(map(lambda i, d: kp[i] if d < MAX_DESC_MSE else None, indices, diffs))
    for i in range(len(indices)):
        ind = indices[i]
        diff = diffs[i]
        if diff > MAX_DESC_MSE or i not in coords.reference_mapping:
            kpts.append(None)
            continue
        kpts.append(kp[ind])


    #img = cv.drawKeypoints(img, kpts, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for i in range(len(kpts)):
        kpt = kpts[i]
        if kpt is None:
            continue
        point = tuple(np.int32(kpt.pt))
        cv.circle(img, point, int(kpt.size / 2), (255, 0, 0), 2)
        cv.putText(img, str(i), tuple(np.int32(point) + [7, 7]), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))

    points = []
    points_real = []
    for i in range(len(kpts)):
        kpt = kpts[i]
        if kpt is None:
            continue
        points.append(kpt.pt)
        points_real.append(coords.map_points[coords.reference_mapping[i]])

    print(points)
    print(points_real)
    if len(points) >= 4:
        Mat = calc_from_cam_to_map_matrix(points[:4], points_real[:4])
        print(Mat)

    cv.imshow("Img", img)
    if cv.waitKey() == ord('q'):
        break
