import pickle 
import cv2
import os
import sys
import argparse
from cv2 import undistort
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from regression import Regressor
from object_detection import ObjectDetector
from calibration import Calibrator
def parse_args():
    parser =argparse.ArgumentParser(description="train calibration and regression")
    #chessboard corner detection and params
    parser.add_argument('--chessboard_corner_re_detect', type=bool, default=False)
    parser.add_argument('--chessboard_image', type=str, help = 'path to chessboard image to corner detection', default='./datasets/chessboard_images/dataset_29072021.jpg')
    parser.add_argument('--origin', nargs="+", help='coordinates of origin point by(x,y) in chessboard image', default=['583','30'])
    parser.add_argument('--cell_length', help= 'edge of a cell of the chessboard in cm', type=int, default = 3)
    #object detection and localization params
    parser.add_argument('--train_object_detection', type = bool, default=False)
    """



    Code here
    
    
    
    """
    #calibration and params
    parser.add_argument('--train_calibrator', type=bool, default= False)
    parser.add_argument('--corners', type=str, default="/")
    #regresson and params
    parser.add_argument('--train_regressor', type=bool, default=True)
    parser.add_argument('--data', type =str, default='./datasets/coordinates/dataset_29072021.csv')
    parser.add_argument('--object_images', type=str, help='path to images', default= './datasets/images/dataset_29072021/')
    parser.add_argument('--out_dir', type=str, default="./out_dir")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    if args.train_object_detector:
        obDetector = ObjectDetector(args)
        obDetector.fit()
    if args.train_calibrator:
        calibrator = Calibrator(args)
        calibrator.fit()
    if args.train_regressor:
        regressor = Regressor(args)
        regressor.fit()
    print("train complete")

if __name__ == "__main__":
    main()  
