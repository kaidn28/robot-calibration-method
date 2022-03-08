import numpy as np
import math
from .functions import *

class Interpolator:
    def __init__(self):
        pass
    def predict(self, point, origin, img_cell_lengths, cell_length, corners_mat):
        # print(f"point: {point}")
        # print(f"origin: {origin}")
        # print(f"img_cell_length: {img_cell_lengths}")
        # print(f"cell_length: {cell_length}")
        # print(corners_mat.shape)
        # print("corners mat: {}".format(corners_mat[:, :, 0]))
        topk = topKNearest(point, corners_mat)
        topk_real = toReal(topk, origin, img_cell_lengths, cell_length)
        # print(topk)
        # print(topk_real)
        #print(np.multiply(point - origin, 1/img_cell_lengths)*cell_length)
    
        int_loc = interpolate(point, topk, topk_real)
        # print(int_loc.shape)
        # return np.multiply(point - origin, 1/img_cell_lengths)*cell_length
        # return np.multiply(point - origin, 1/img_cell_lengths)*cell_length
        return int_loc
        
