import numpy as np

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

def calc_from_cam_to_map_matrix(centers, marker_coords):
    #use ray match data as linear equations
    A1 = np.zeros((N_MARKERS * 2, 4))
    A2 = np.zeros((N_MARKERS * 2, 4))
    A3 = np.zeros((N_MARKERS * 2, 4))

    A1[::2, :] = marker_coords.T

    A2[::2, :] = marker_coords.T
    A2 = np.roll(A2, 1, axis=0)

    A3[::2, :] = -marker_coords.T * centers[1:2].T
    A3 = np.roll(A3, 1, axis=0)
    A3[::2, :] = -marker_coords.T * centers[0:1].T

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

