import numpy as np
import cv2 as cv
import os
import yaml

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def gauss(A, stop_j=None, stop_i=None, start_i=0, start_j=0, eps=1e-6):
    A = A.copy()
    i = start_i
    j = start_j
    n = A.shape[0]
    m = A.shape[1]
    if stop_j is None:
        stop_j = m
    if stop_i is None:
        stop_i = n
    while i < stop_i and j < stop_j:
        if np.abs(A[i, j]) < eps:
            A[i, j] = 0
        if A[i, j] == 0:
            i1 = np.nonzero(A[i:, j])
            if len(i1[0]) == 0:
                j += 1
                continue
            else:
                #i1 = i1[0][0] + i
                i1 = np.argmax(np.abs(A[i:, j])) + i
                if np.abs(A[i1, j]) < eps:
                    A[i1, j] = 0
                    j += 1
                    continue
            tmp = A[i].copy()
            A[i] = A[i1]
            A[i1] = tmp
        A[i] = A[i] / A[i, j]
        A[i + 1:] -= A[i, :] * A[i + 1:, j:j + 1]
        A[:i] -= A[i, :] * A[:i, j:j + 1]
        i += 1
        j += 1

    return A


def find_in_big(small, big):
    small_rpt = np.repeat([small], len(big), axis=0)
    big_rpt = np.repeat([big], len(small), axis=0)
    big_rpt = np.transpose(big_rpt, (1, 0, 2))

    indices = np.zeros((len(small),), dtype=np.int32)
    mses = np.zeros((len(small),), dtype=np.float32)
    ind_lookup = np.linspace(0, len(small) - 1, len(small), dtype=np.int32)
    k_ind_lookup = np.linspace(0, len(big) - 1, len(big), dtype=np.int32)

    diffs = np.average(np.square(big_rpt - small_rpt), axis=-1)
    for i in range(len(small)):
        k, j = np.divmod(np.argmin(diffs), diffs.shape[1])
        diff = diffs[k, j]
        j1 = ind_lookup[j]
        k1 = k_ind_lookup[k]
        diffs = np.delete(diffs, j, axis=1)
        diffs = np.delete(diffs, k, axis=0)
        ind_lookup = np.delete(ind_lookup, j, axis=0)
        k_ind_lookup = np.delete(k_ind_lookup, k, axis=0)
        indices[j1] = k1
        mses[j1] = diff / 255 / 255

    return indices, mses


def calc_from_cam_to_map_matrix(cam_real_coords, map_real_coords, newmat):
    N_MARKERS = len(cam_real_coords)

    cam_real_coords = np.float32(cam_real_coords)
    map_real_coords = np.float32(map_real_coords)

    cam_real_coords = np.dot(np.linalg.inv(newmat), np.vstack([cam_real_coords.T, np.ones((1, N_MARKERS))])).T

    map_real_coords = np.hstack([map_real_coords, np.ones((N_MARKERS, 1))]).T
    cam_real_coords = np.hstack([cam_real_coords, np.ones((N_MARKERS, 1))]).T

    # print(map_real_coords.shape)
    # print(cam_real_coords.shape)

    # use ray match data as linear equations
    # A1 = np.zeros((N_MARKERS * 2, 4))
    # A2 = np.zeros((N_MARKERS * 2, 4))
    # A3 = np.zeros((N_MARKERS * 2, 4))
    A = np.zeros((N_MARKERS * 2, 12))

    A[:N_MARKERS, 0:4] = map_real_coords.T
    A[:N_MARKERS, 8:12] = -map_real_coords.T * cam_real_coords[0:1].T

    A[N_MARKERS:, 4:8] = map_real_coords.T
    A[N_MARKERS:, 8:12] = -map_real_coords.T * cam_real_coords[1:2].T

    # first we need to pre-solve system so that we can add normality equations
    # simplify system and use first 4 points (11 equations)
    Ag = gauss(A, stop_j=4)
    Ag = gauss(Ag, start_i=0, stop_j=11)
    Ag = Ag[:11]

    np.set_printoptions(2, suppress=True, linewidth=500)
    # print(Ag)

    if np.argmax(np.abs(Ag[:, 3])) < 2:
        raise Exception("Points are on the same line. Can't calculate matrix")
    elif np.argmax(np.abs(Ag[:, 3])) == 2:
        if (np.count_nonzero(Ag[0, :3]) > 1) or (np.count_nonzero(Ag[1, :3]) > 1):
            raise Exception("Can't extract points. They are on the same plane and wtf")

        # need to add orthogonality to the system
        for j in range(0, 3):
            if (np.count_nonzero(Ag[0:2, j]) == 0):
                jz = j

        Aorth = np.zeros((3, 12))

        a1 = np.float32([Ag[0, -1], Ag[3, -1], Ag[6, -1]])
        a2 = np.float32([Ag[1, -1], Ag[4, -1], Ag[7, -1]])
        a3 = np.float32([
            a1[1] * a2[2] - a1[2] * a2[1],
            -a1[0] * a2[2] + a1[2] * a2[0],
            a1[0] * a2[1] - a1[1] * a2[0]
        ])
        a3 = a3 / np.sqrt(np.sqrt(np.dot(a1, a1)) * np.sqrt(np.dot(a2, a2)))

        Aorth[0, jz] = 1
        Aorth[0, -1] = -a3[0]
        Aorth[1, 4 + jz] = 1
        Aorth[1, -1] = -a3[1]
        Aorth[2, 8 + jz] = 1
        Aorth[2, -1] = -a3[2]

        Ag[-3:] = Aorth
        Ag = gauss(Ag, stop_j=11)
        # print(Ag)

    # now add normality to the system
    a = 1 / np.sqrt(np.dot(Ag[0:3, 11], Ag[0:3, 11]))
    Ag = np.hstack([Ag, np.zeros((11, 1))])
    Ag = np.vstack([Ag, [0,0,0,0, 0,0,0,0, 0,0,0,1, -a]])
    Ag = gauss(Ag)
    # print(Ag)

    A = np.vstack([Ag[:,:12], A])
    b = np.vstack([-Ag[:, -1:], np.zeros((N_MARKERS * 2, 1))])

    ATA = np.dot(A.T, A)
    ATb = np.dot(A.T, b)

    ATA_ATb = np.hstack([ATA, ATb])

    Ag = gauss(ATA_ATb, stop_j=12)
    # print(Ag)

    # system is solved, we can gather matrix elements
    M = Ag[:, -1]
    M = np.reshape(M, (3, 4))

    # normalizing matrix for future use (it is already "almost" normalized, but with some error)
    M[:, 0] /= np.sqrt(np.dot(M[:, 0], M[:, 0]))
    M[:, 1] /= np.sqrt(np.dot(M[:, 1], M[:, 1]))
    M[:, 2] /= np.sqrt(np.dot(M[:, 2], M[:, 2]))

    # inverting matrix to have from cam to map mat.
    M = np.linalg.inv(np.vstack([M, [0, 0, 0, 1]]))[:3]
    return M

def calc_from_cam_to_map_matrix_not_bullshit(cam_pos, cam_pts, map_pts, newmat):
    cam_pos = np.float32(cam_pos)
    if len(cam_pos.shape) == 1:
        cam_pos = np.float32([cam_pos]).T
    else:
        cam_pos  = cam_pos.T

    N_MARKERS = min(len(cam_pts), len(map_pts))

    cam_pts = np.float32(cam_pts)
    cam_pts = np.dot(np.linalg.inv(newmat), np.vstack([cam_pts.T, np.ones((1, N_MARKERS))])).T

    cam_pts = np.float32(cam_pts).T
    map_pts = np.float32(map_pts).T

    N_POINTS = min(cam_pts.shape[1], map_pts.shape[1])

    # Normalizing rays directions
    map_pts = map_pts - cam_pos
    cam_pts /= np.sqrt(np.sum(np.square(cam_pts), axis=0))
    map_pts /= np.sqrt(np.sum(np.square(map_pts), axis=0))

    # Building equation system
    A = np.zeros((N_POINTS * 3, 9))
    b = np.zeros((N_POINTS * 3, 1))

    for i in range(N_POINTS):
        cam_pt = cam_pts[:,i]
        map_pt = map_pts[:,i]
        A[i*3 + 0, 0:3] = cam_pt
        A[i*3 + 1, 3:6] = cam_pt
        A[i*3 + 2, 6:9] = cam_pt
        b[i*3:(i+1)*3, 0] = map_pt

    ATA = np.dot(A.T, A)
    ATb = np.dot(A.T, b)

    x = np.linalg.solve(ATA, ATb)
    M = np.reshape(x, (3, 3))
    M = np.hstack([M, cam_pos])

    return M



if __name__ == '__main__':
    np.set_printoptions(2, suppress=True, linewidth=500)

    N_POINTS = 20

    alpha = np.pi / 4

    cam_pos = [-1, -2, -3]

    # M = np.float32([
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 0]
    # ])

    M = np.float32([
        [np.cos(alpha), -np.sin(alpha), 0, cam_pos[0]],
        [np.sin(alpha), np.cos(alpha), 0, cam_pos[1]],
        [0, 0, 1, cam_pos[2]]
    ])
    # R = np.float32([
    #         [np.cos(alpha), -np.sin(alpha), 0],
    #         [np.sin(alpha), np.cos(alpha), 0],
    #         [0, 0, 1]
    #     ])
    # M = np.dot(R, np.roll(np.roll(R, 1, 0), 1, 1))
    # M = np.dot(R, np.roll(np.roll(R, 1, 0), 1, 1))
    # M = np.hstack([M, np.transpose([cam_pos])])
    M_1 = np.linalg.inv(np.vstack([M, [0,0,0,1]]))[:3]

    print(M)



    points_map = np.random.normal(loc=0, size=(N_POINTS, 3))
    # points_map[:,1] = 0
    points_cam = np.dot(M_1, np.vstack([points_map.T, np.ones((1, N_POINTS))]))
    points_cam += np.random.normal(size=(3, N_POINTS), scale=0.01)
    points_cam /= points_cam[2]
    points_cam = points_cam.T

    M2 = calc_from_cam_to_map_matrix_not_bullshit(cam_pos, points_cam, points_map)
    M3 = calc_from_cam_to_map_matrix(points_cam, points_map)
    print(M2)
    print(M3)
