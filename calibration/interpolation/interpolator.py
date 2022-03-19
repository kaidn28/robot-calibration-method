import numpy as np
import math
from .functions import *

class Interpolator:
    def __init__(self):
        pass
    def predict(self, point, origin, img_cell_lengths, cell_length, corners_mat):
        topk = topKNearest(point, corners_mat)
        # print('topk')
        # print(topk)
        topk_real = toReal(topk, origin, img_cell_lengths, cell_length)
        # print("top k real: ", topk_real)
        int_loc1 = interpolate_linear(point, topk, topk_real)
        # print("int_loc1")
        # print(int_loc1)
        int_loc = np.zeros((2))
        nearest_corners = nearest4Ways(point, corners_mat)
        # print(nearest_corners)
        # print("nearest_corners: \n", nearest_corners)
        nearest_corners_real = toReal(nearest_corners, origin, img_cell_lengths, cell_length)
        # print("nearest_corners_real: \n", nearest_corners_real)
        # print(nearest_corners_real)
        int_loc = interpolate_bilinear(point, nearest_corners, nearest_corners_real)
        # print('int_loc1')
        # print(int_loc1)
        return np.multiply(point - origin, 1/img_cell_lengths)*cell_length, int_loc1, int_loc 
        
