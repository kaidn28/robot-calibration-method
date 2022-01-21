import pickle 
import cv2
import os
import sys
import argparse
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from regression import Regressor
def parse_args():
    parser =argparse.ArgumentParser(description="train calibration and regression")
    parser.add_argument('train-chessboard-corner-detector', type=bool, default=False)
    parser.add_argument('--chessboard-image', type=str, help = 'path to chessboard image to corner detection', default='./datasets/chessboard_images/dataset_29072021.jpg')
    parser.add_argument('--origin', nargs="+", help='coordinates of origin point by(x,y) in chessboard image', default=['583','30'])
    parser.add_argument('train-regressor', type=bool, default=False)
    parser.add_argument('--data', type =str, default='./datasets/coordinates/dataset_29072021.csv')
    parser.add_argument('--object-images', type=str, help='path to images', default= './datasets/images/dataset_29072021/')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    reg = Regressioner(args)
    print(args)

if __name__ == "__main__":
    main()  