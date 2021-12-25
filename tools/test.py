import pickle 
import cv2
import os
import sys
import argparse
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

def parse_args():
    parser =argparse.ArgumentParser(description="Run calibration")
    parser.add_argument('--images', type=str, help='path to images', default= './datasets/images/dataset_29072021/')
    parser.add_argument('--origin', nargs="+", help='coordinates of origin point by(x,y)', default=['583','30'])
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)

if __name__ == "__main__":
    main()