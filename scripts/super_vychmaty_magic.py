#!/usr/bin/python3

import numpy as np

import tensorflow as tf
tf.enable_eager_execution()

def loss(A, x, u):
    u1 = tf.linalg.matmul(A, x)
    u1 /= u1[2]

    diff = u - u1
    diff = tf.square(diff)
    diff = tf.reduce_mean(diff)

    diff = tf.stack([12 * diff,  
        tf.tensordot(A[:,1], A[:,1], 1) - 1,
        tf.tensordot(A[:,1], A[:,2], 1),
        tf.tensordot(A[:,1], A[:,3], 1),
        tf.tensordot(A[:,2], A[:,2], 1) - 1,
        tf.tensordot(A[:,2], A[:,3], 1),
        tf.tensordot(A[:,3], A[:,3], 1) - 1,
    ])

    diff = tf.square(diff)
    diff = tf.reduce_mean(diff)

    return diff

def calc_optimal_matrix(x, u, maxLoss=0.001, maxEpochs=100):
    A = tf.eye(3, 4)
    A = tf.Variable(A, trainable=True)

    x = tf.constant(x, dtype=tf.float32)
    u = tf.constant(u, dtype=tf.float32)

    optimizer = tf.train.AdamOptimizer(0.01)

    for epoch in range(maxEpochs):
        with tf.GradientTape() as tape:
            L = loss(A, x, u)

        grad = tape.gradient(L, A)
        optimizer.apply_gradients([(grad, A)])
        A1 = A.value().numpy()
        A1[:,:3] /= np.sum(np.square(A[:,:3]), axis=0)
        A.assign(A1)

        if L < maxLoss:
            break

        print(epoch, L.numpy())

    A = tf.reshape(A, (3, 4)).numpy()

    return A

if __name__ == "__main__":

    A = np.float32([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3]
        ])

    N_POINTS = 10

    x = np.random.normal(size=(3, N_POINTS))
    x[2] += 5
    x = np.vstack([x, np.ones((1, N_POINTS))])

    u = np.dot(A, x)
    u /= u[2]

    A1 = calc_optimal_matrix(x, u)

    np.set_printoptions(2, suppress=True)
    print(A1)

