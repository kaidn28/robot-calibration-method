import pickle 
import cv2
import os
import sys
import argparse
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from regression import Regressor
def parse_args():
    parser =argparse.ArgumentParser(description="train calibration and regression")
    
    #regresson and params
    parser.add_argument('--train_regressor', type=bool, default=True)
    parser.add_argument('--data', type =str, default='./datasets/coordinates/dataset_29072021.csv')
    parser.add_argument('--object_images', type=str, help='path to images', default= './datasets/images/dataset_29072021/')
    parser.add_argument('--out_dir', type=str, default="./out_dir/train/regressor/parameters/")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    if args.train_regressor:
        regressor = Regressor(args)
        regressor.fit()
    print("train complete")

if __name__ == "__main__":
    main()  
