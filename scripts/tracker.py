#!/usr/bin/python3

import cv2 as cv
import os
import yaml
import numpy as np
import glob
import time
import json

from utils import calc_from_cam_to_map_matrix
from processing import grey_world, Sobel_3_colors, correct_distortion, preprocess, sift
import processing
import utils
import coords


MAX_DESC_MSE = 0.007
MIN_QUALITY = 0.0

with open("keypoints.json", "rt") as kpfile:
    points_good = json.load(kpfile)
des_good = list(map(lambda x: x["desc"], points_good))

for file in glob.glob("../test_images/*.jpg") + ["../test_images/test_img.png"] + ["../test_images/Image.png"]:
    img = cv.imread(file)

    img = preprocess(img)

    kp, des = sift.detectAndCompute(img, None)
    kpdes = list(zip(kp, des))

    indices, diffs = utils.find_in_big(des_good, des)

    #img = cv.drawKeypoints(img, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kpts = []#list(map(lambda i, d: kp[i] if d < MAX_DESC_MSE else None, indices, diffs))
    for i in range(len(indices)):
        ind = indices[i]
        diff = diffs[i]
        if diff > MAX_DESC_MSE or i not in coords.reference_mapping:
            kpts.append(None)
            continue
        kpts.append(kp[ind])

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
        point = coords.reference_mapping[i]
        points_real.append(point)

    if len(points) >= 6:
        #Mat = utils.calc_from_cam_to_map_matrix_not_bullshit([1.5, 2, 1], points, points_real, processing.newmat)

        ret, rvec, tvec = cv.solvePnP(np.float32(points_real), np.float32(points), processing.camera_matrix, processing.distCoeffs)
        
        R = cv.Rodrigues(rvec)[0]

        A = np.hstack([R, tvec])
        
        A = np.vstack([A, [0, 0, 0, 1]])
        A = np.linalg.inv(A)[:3]

        np.set_printoptions(3, suppress=True)
        print(A)

    cv.imshow("Img", img)
    if cv.waitKey() == ord('q'):
        break
