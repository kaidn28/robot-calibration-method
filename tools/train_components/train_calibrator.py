import pickle 
import cv2
import os
import sys
import argparse
from cv2 import undistort
import pandas as pd
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from calibration import Calibrator
def parse_args():
    parser =argparse.ArgumentParser(description="train calibration and regression")
    #chessboard corner detection and params
    parser.add_argument('--chessboard_corner_re_detect', type=bool, default=False)
    parser.add_argument('--chessboard_image', type=str, help = 'path to chessboard image to corner detection', default='./datasets/chessboard_images/dataset_29072021.jpg')
    parser.add_argument('--cell_length', help= 'edge of a cell of the chessboard in cm', type=int, default = 3)
    #image correction and params
    parser.add_argument('--train_image_correction', type=bool, default= False)
    parser.add_argument('--corners', type=str, default="/")
    parser.add_argument('--out_dir', type=str, default="./out_dir/train/calibration/")
    args = parser.parse_args()
    #origin detection
    parser.add_argument('--train_origin_detection', type=bool, default= False)
    

    
    return args

def main():
    args = parse_args()
    # print(args)
    calibrator = Calibrator()
    calibrator.fit(args)

    

if __name__ == "__main__":
    main()  
