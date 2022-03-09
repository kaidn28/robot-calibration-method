import numpy as np
import math
from .functions import *

class Interpolator:
    def __init__(self):
        pass
    def predict(self, point, origin, img_cell_lengths, cell_length, corners_mat, method = "bilinear"):
        if method == "linear":
            topk = topKNearest(point, corners_mat)
            topk_real = toReal(topk, origin, img_cell_lengths, cell_length)
        
            int_loc = interpolate_linear(point, topk, topk_real)
        elif method == "bilinear":
            
            nearest_corners = nearest4Ways(point, corners_mat)
            print("nearest_corners: \n", nearest_corners.reshape(2,4))
            nearest_corners_real = toReal(nearest_corners, origin, img_cell_lengths, cell_length)
            # print(nearest_corners)

            print("nearest_corners_real: \n", nearest_corners_real.reshape(2,4))
            # print(nearest_corners_real)
            int_loc = interpolate_bilinear(point, nearest_corners, nearest_corners_real)
            # print(int_loc)
        return int_loc, np.multiply(point - origin, 1/img_cell_lengths)*cell_length
        
