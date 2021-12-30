import pickle 
import cv2
import os
import sys
import argparse
from numpy import integer
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from chessboard_corner_detection import *
def parse_args():
    parser =argparse.ArgumentParser(description="Run calibration")
    parser.add_argument('--chessboard-image', default="./chessboard_corner_detection/chessboard.jpg")
    parser.add_argument('--images', type=str, help='path to images', default= './datasets/images/dataset_29072021/')
    parser.add_argument('--origin', nargs="+", help='coordinates of origin point by(x,y)', default=['583','30'])
    parser.add_argument('--chessboard_cell_edge_length', help= 'length of a cell in chessboard in cm', type=integer, default = 3)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    cDetector = ChessboardCornerDetector()
    cDetector.detect(args)

    

if __name__ == "__main__":
    main()