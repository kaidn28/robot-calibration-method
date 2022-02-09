import pickle 
import cv2
import os
import sys
import argparse
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from regression import Regressor
from chessboard_corner_detection import ChessboardCornerDetector
def parse_args():
    parser =argparse.ArgumentParser(description="train calibration and regression")
    parser.add_argument('--chessboard_corner_re_detect', type=bool, default=False)
    parser.add_argument('--chessboard_image', type=str, help = 'path to chessboard image to corner detection', default='./datasets/chessboard_images/dataset_29072021.jpg')
    parser.add_argument('--origin', nargs="+", help='coordinates of origin point by(x,y) in chessboard image', default=['583','30'])
    parser.add_argument('--cell_length', help= 'edge of a cell of the chessboard in cm', type=int, default = 3)
    parser.add_argument('--train_regressor', type=bool, default=True)
    parser.add_argument('--data', type =str, default='./datasets/coordinates/dataset_29072021.csv')
    parser.add_argument('--object_images', type=str, help='path to images', default= './datasets/images/dataset_29072021/')
    parser.add_argument('--out_dir', type=str, default="./out_dir")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    if args.chessboard_corner_re_detect:
        cc_detector = ChessboardCornerDetector(args)
        cc_detector.train()
    if args.train_regressor:
        regressor = Regressor(args)
        regressor.train()


if __name__ == "__main__":
    main()  
