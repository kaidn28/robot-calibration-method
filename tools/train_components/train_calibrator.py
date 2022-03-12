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
    parser.add_argument('--chessboard_image', type=str, help = 'path to chessboard image to corner detection', default='./datasets/chessboard_images/dataset_05032022.png')
    parser.add_argument('--cell_length', help= 'edge of a cell of the chessboard in cm', type=int, default = 2.9)
    #image correction and params
    parser.add_argument('--train_image_correction', type=bool, default= False)
    parser.add_argument('--corners', type=str, default="/")
    parser.add_argument('--out_dir', type=str, default="./out_dir/")
    
    #origin detection
    parser.add_argument('--train_origin_detection', type=bool, default= False)
    
    #mode 
    parser.add_argument('--calibration_mode', type=str, default="train")
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    # print(args)
    calibrator = Calibrator(args)
    img = cv2.imread(args.chessboard_image)
    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    calibrator.fit(img)

    

if __name__ == "__main__":
    main()  
