#!/usr/bin/python3

import numpy as np

import tensorflow as tf
tf.enable_eager_execution()

import sys

def rotMat(rx, ry, rz):
    Rx = tf.stack([
        [1, 0, 0],
        [0, tf.cos(rx), -tf.sin(rx)],
        [0, tf.sin(rx), tf.cos(rx)],
        ])

    Ry = tf.stack([
        [tf.cos(ry), 0, -tf.sin(ry)],
        [0, 1, 0],
        [tf.sin(ry), 0, tf.cos(ry)],
        ])

    Rz = tf.stack([
        [tf.cos(rz), -tf.sin(rz), 0],
        [tf.sin(rz), tf.cos(rz), 0],
        [0, 0, 1],
        ])

#    R0 = tf.constant([
#        [-1.0, 0.0, 0.0],
#        [0.0,  1.0/np.sqrt(2), -1.0/np.sqrt(2)],
#        [0.0, -1.0/np.sqrt(2), -1.0/np.sqrt(2)],
#        ])

    R0 = tf.constant([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        ])

    R = tf.matmul(Ry, R0)
    R = tf.matmul(Rx, R)
    R = tf.matmul(Rz, R)

    return R

def loss(rx, ry, rz, x0, x, u):

    R = rotMat(rx, ry, rz)

    u1 = tf.linalg.matmul(R, tf.transpose(tf.transpose(x) - tf.transpose(x0)))
    u1 /= u1[2]

    diff = u - u1
    diff = tf.square(diff)
    diff = tf.reduce_mean(diff)

    return diff

def calc_optimal_matrix(x, u, maxLoss=0.001, maxEpochs=1000):
    rx = tf.Variable(1.0, trainable=True, dtype=tf.float32)
    ry = tf.Variable(1.0, trainable=True, dtype=tf.float32)
    rz = tf.Variable(1.0, trainable=True, dtype=tf.float32)
    x0 = tf.Variable(tf.zeros((3, 1), dtype=tf.float32), trainable=True)

    x = tf.constant(x, dtype=tf.float32)
    u = tf.constant(u, dtype=tf.float32)

    optimizer = tf.train.AdamOptimizer(0.001)

    for epoch in range(maxEpochs):
        with tf.GradientTape() as tape:
            L = loss(rx, ry, rz, x0, x, u)

        grad = tape.gradient(L, [rx, ry, rz, x0])
        optimizer.apply_gradients(zip(grad, [rx, ry, rz, x0]))

        if L < maxLoss:
            break

        sys.stdout.write("\r" + str(epoch) + " " + str(L.numpy()))
        sys.stdout.flush()

    print(rx, ry, rz)

    R = rotMat(rx, ry, rz)

    return R.numpy(), x0.value().numpy()

if __name__ == "__main__":

    R = np.float32([
            [1/np.sqrt(2), -1/np.sqrt(2), 0],
            [1/np.sqrt(2), 1/np.sqrt(2), 0],
            [0, 0, 1]
        ])

    x0 = np.float32([
            [0],
            [0],
            [0]
        ])

    A = [ -8.9867124231203199e-01, 4.5699009049497752e-02,
       -9.5565323713849515e-02, 1.5000000000000004e+00,
       -3.1834273002864288e-02, 6.3023537145045461e-01,
       -5.9322467897135334e-01, 2., -4.7854151524381980e-02,
       -5.7879190767273969e-01, -7.2088708701779647e-01,
       1.0199999809265139e+00 ]
    
    A = np.reshape(A, (3, 4))
    R = A[:,:3]
    x0 = A[:,3:4]

    N_POINTS = 10

    x = np.random.normal(size=(3, N_POINTS))
#    x = np.vstack([x, np.ones((1, N_POINTS))])

    u = np.dot(R, (x.T - x0.T).T)
    u /= u[2]



    print(R)
    print(x0)

    R1, x01 = calc_optimal_matrix(x, u)
    np.set_printoptions(2, suppress=True)

    print(R)
    print(x0)

    print(R1)
    print(x01)

