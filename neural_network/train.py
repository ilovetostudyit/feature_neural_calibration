#!/usr/bin/python3

import tensorflow as tf
tf.enable_eager_execution()
import glob
import pypdn
import cv2 as cv
import numpy as np
import sys
import pickle


WIDTH = 640
HEIGHT = 480
SIZE = (WIDTH, HEIGHT)

OUT_WIDTH = 33
OUT_HEIGHT = 23


def load_dataset():
    X_train = []
    Y_train = []
    cnt = 0
    all_len = len(glob.glob("../marked_up/*.pdn"))
    for file in glob.glob("../marked_up/*.pdn"):
        pdnimg = pypdn.read(file)
        
        img = pdnimg.layers[0].image
        img = np.float32(img / 255)
        img = cv.resize(img, SIZE)
        
        confmaps = list(map(lambda layer: layer.image[3], pdnimg.layers[1:]))
        confmaps = np.float32(confmaps) / 127.5 - 1
        if (confmaps.shape[0] != 24):
            confmaps = np.vstack([confmaps, np.zeros_like(confmaps[0:1])])

        confmaps = np.transpose(confmaps, (1, 2, 0))
        confmaps = cv.resize(confmaps, (OUT_WIDTH, OUT_HEIGHT))

        X_train.append(img)
        Y_train.append(confmaps)

        sys.stdout.write("\r{} / {}".format(cnt, all_len))
        sys.stdout.flush()
        cnt += 1

    X_train = np.float32(X_train)
    Y_train = np.float32(Y_train)

    return X_train, Y_train

def Extractor():
    inp = tf.keras.layers.Input((HEIGHT, WIDTH, 4))
    x = inp

    x = tf.keras.layers.Convolution2D(8, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Convolution2D(8, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)

    x = tf.keras.layers.Convolution2D(16, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Convolution2D(16, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    
    x = tf.keras.layers.Convolution2D(32, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Convolution2D(32, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Convolution2D(32, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    
    x = tf.keras.layers.Convolution2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Convolution2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Convolution2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Convolution2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)

    x = tf.keras.layers.Convolution2D(24, (3, 3), activation='relu', padding='same')(x)
    
    return tf.keras.Model(inputs=inp, outputs=x)

#X_train, Y_train = load_dataset()
#pickle.dump((X_train, Y_train), open("dataset.pickle", "wb"))
#exit()

X_train, Y_train = pickle.load(open("dataset.pickle", "rb"))

model = Extractor()

model.compile(loss='mae', optimizer='adam')

model.fit(X_train, Y_train, epochs=10)
model.save("model.h5")



