import numpy as np
import math 

def topKNearest(point, mat, k = 4):
    shape = mat.shape
    assert len(shape) == 3 and shape[2] == 2
    min_ini = np.array([-9999, -9999])
    topk = [min_ini for i in range(k)]
    for x in range(shape[0]):
        for y in range(shape[1]):
            cell = mat[x, y]
            dis = np.linalg.norm(cell - point)
            min = np.linalg.norm(topk[k-1] - point)
            if dis < min:
                topk[k-1] = cell
                topk.sort(key=lambda a: np.linalg.norm(a - point))
    return np.array(topk)

def toReal(topk, origin, ic_len, c_len):
    re_allo = topk - origin
    topk_real = roundToPoint5(np.multiply(re_allo, 1/ic_len))*c_len
    # print(a2)
    return topk_real

def roundToPoint5(num):
    return np.round(num + 0.5) - 0.5

def getLRParams(topk, topk_real):
    bias_vec = np.ones((topk.shape[0], 1))
    topk_hat = np.concatenate((topk, bias_vec), axis = 1)
    inv_mat = np.linalg.inv(topk_hat.T.dot(topk_hat))
    # print(inv_mat.shape)
    w = inv_mat.dot(topk_hat.T).dot(topk_real)
    return w

def interpolate(point, topk, topk_real):
    point = np.append(point, 1).reshape(1, -1)
    point = point.reshape(1, -1)
    w = getLRParams(topk, topk_real)
    int_point = point.dot(w)
    # print(int_point)
    return int_point.reshape(-1)