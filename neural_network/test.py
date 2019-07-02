#!/usr/bin/python3

import cv2 as cv
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np

WIDTH = 640
HEIGHT = 480

img = cv.imread("../test_images/2019-03-02-150600.jpg") 
img = cv.cvtColor(img, cv.COLOR_BGR2RGBA)
img = cv.resize(img, (WIDTH, HEIGHT))

model = tf.keras.models.load_model("./model.h5")

maps = model(np.float32([img / 255.0]))[0].numpy()
maps = maps * 0.5 + 0.5

print(maps.shape)
for i in range(24):
    map = maps[:,:,i]
    print(map)
    cv.imshow("Map", maps[:,:,i])
    cv.waitKey()

