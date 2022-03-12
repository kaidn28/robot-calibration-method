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


def nearest4Ways(p, mat):
    shape = mat.shape
    assert len(shape) == 3 and shape[2] == 2
    lt = None
    rt = None
    lb = None 
    rb = None
    lt_dis = 9999
    rt_dis = 9999
    lb_dis = 9999
    rb_dis = 9999
    for x in range(shape[0]):
        for y in range(shape[1]):
            c = mat[x,y]
            if np.linalg.norm(c) == 0:
                continue

            dis = np.linalg.norm(p - c)
            if c[0] < p[0] and c[1] < p[1] and dis < lt_dis:
                lt = c
                lt_dis = dis
            elif c[0] > p[0] and c[1] < p[1] and dis < rt_dis:
                rt = c
                rt_dis = dis 
            elif c[0] < p[0] and c[1] > p[1] and dis < lb_dis:
                lb = c
                lb_dis = dis
            elif c[0] > p[0] and c[1] > p[1] and dis < rb_dis:
                rb = c
                rb_dis = dis
    return np.array([lt, rt, lb, rb])

def toReal(topk, origin, ic_len, c_len):
    print("topk : ")
    print(topk)
    print("origin: ")
    print(origin)
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

def interpolate_linear(point, topk, topk_real):
    point = np.append(point, 1).reshape(1, -1)
    point = point.reshape(1, -1)
    w = getLRParams(topk, topk_real)
    int_point = point.dot(w)
    # print(int_point)
    return int_point.reshape(-1)

def interpolate_bilinear(p, nc, nc_real):
    # print("point: ", p)
    x, y = p
    lt, rt, lb, rb = nc
    real_lt, real_rt, real_lb, real_rb = nc_real
    
    top_frac = (x- lt[0])/(rt[0]-lt[0])
    bot_frac = (x - lb[0])/(rb[0]-lb[0])
    top = (x, lt[1] + top_frac*(rt[1] - lt[1]))
    real_top = (real_lt[0] + top_frac*(real_rt[0] - real_lt[0]), real_lt[1] + top_frac*(real_rt[1] - real_lt[1]))
    
    bot = (x, lb[1] + bot_frac*(rb[1] - lb[1]))
    real_bot = (real_lb[0] + bot_frac*(real_rb[0]- real_lb[0]), real_lb[1] + top_frac*(real_rb[1] - real_lb[1]))

    tb_frac = (y-top[1])/(bot[1]-top[1])
    real_x = real_top[0]+ tb_frac*(real_bot[0]-real_top[0])
    real_y = real_top[1] + tb_frac*(real_bot[1] - real_top[1])
    return np.array([real_x, real_y])
