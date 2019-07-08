import numpy as np
import cv2 as cv
import os
import yaml

def gauss(A):
    A = A.copy()
    i = 0
    j = 0
    n = A.shape[0]
    m = A.shape[1]
    while i < n and j < m:
        if A[i, j] == 0:
            i1 = np.nonzero(A[i:,j])
            if len(i1[0]) == 0:
                j += 1
                continue
            i1 = i1[0][0] + i
            tmp = A[i].copy()
            A[i] = A[i1]
            A[i1] = tmp
        A[i] = A[i] / A[i, j]
        A[i+1:] -= A[i, :] * A[i+1:,j:j+1]
        A[:i] -= A[i, :] * A[:i,j:j+1]
        i += 1
        j += 1

    return A

def calc_from_cam_to_map_matrix(cam_real_coords, map_real_coords):
    #use ray match data as linear equations
    A1 = np.zeros((N_MARKERS * 2, 4))
    A2 = np.zeros((N_MARKERS * 2, 4))
    A3 = np.zeros((N_MARKERS * 2, 4))

    A1[::2, :] = map_real_coords.T

    A2[::2, :] = map_real_coords.T
    A2 = np.roll(A2, 1, axis=0)

    A3[::2, :] = -map_real_coords.T * cam_real_coords[1:2].T
    A3 = np.roll(A3, 1, axis=0)
    A3[::2, :] = -map_real_coords.T * cam_real_coords[0:1].T

    A123 = np.hstack([A1, A2, A3])

    # simplify system
    Ag = utils.gauss(A123)
    # np.set_printoptions(2, suppress=True)
    # print(Ag)

    # system needs 4 more equations

    # 2 we take from orthogonality of matrix
    Ag = np.vstack([Ag, [0, 0, Ag[0, -1], 0, 0, 0, Ag[3, -1], 0, 0, 0, Ag[6, -1], 0]])
    Ag = np.vstack([Ag, [0, 0, Ag[1, -1], 0, 0, 0, Ag[4, -1], 0, 0, 0, Ag[7, -1], 0]])
    Ag = utils.gauss(Ag)

    # 1 more from orthogonality
    a23_33 = -np.dot(Ag[4:6, -1], Ag[8:10, -1])
    a23 = -Ag[6, 10]
    a33a = np.sqrt(a23_33 / a23)

    Ag = np.vstack([Ag, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, a33a]])
    Ag = utils.gauss(Ag)
    #print(Ag)

    # and 1 more from normality
    a1 = -Ag[0:3, -1]
    a_2 = 1 / np.dot(a1, a1)
    a = np.sqrt(a_2)

    Ag = np.hstack([Ag, np.zeros((11, 1))])
    Ag = np.vstack([Ag, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, a]])
    Ag = utils.gauss(Ag)
    #print(Ag)

    # system is solved, we can gather matrix elements
    A = Ag[:, -1]
    A = np.reshape(A, (3, 4))

    # normalizing matrix for future use (it is already "almost" normalized, but with some error)
    A[:, 0] /= np.sqrt(np.dot(A[:, 0], A[:, 0]))
    A[:, 1] /= np.sqrt(np.dot(A[:, 1], A[:, 1]))
    A[:, 2] /= np.sqrt(np.dot(A[:, 2], A[:, 2]))

    # inverting matrix to have from cam to map mat.
    A = np.linalg.inv(np.vstack([A, [0, 0, 0, 1]]))[:3]
    return A

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

def find_in_big(small, big):
    small_rpt = np.repeat([small], len(big), axis=0)
    big_rpt = np.repeat([big], len(small), axis=0)
    big_rpt = np.transpose(big_rpt, (1, 0, 2))

    indices = np.zeros((len(small),), dtype = np.int32)
    mses = np.zeros((len(small),), dtype = np.float32)
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
