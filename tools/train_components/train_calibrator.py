import pickle 
import cv2
import os
import sys
import argparse
from cv2 import undistort
import pandas as pd
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from chessboard_corner_detection import ChessboardCornerDetector
from calibration import Calibrator
def parse_args():
    parser =argparse.ArgumentParser(description="train calibration and regression")
    #chessboard corner detection and params
    parser.add_argument('--chessboard_corner_re_detect', type=bool, default=False)
    parser.add_argument('--chessboard_image', type=str, help = 'path to chessboard image to corner detection', default='./datasets/chessboard_images/dataset_29072021.jpg')
    parser.add_argument('--origin', nargs="+", help='coordinates of origin point by(x,y) in chessboard image', default=['583','30'])
    parser.add_argument('--cell_length', help= 'edge of a cell of the chessboard in cm', type=int, default = 3)
    #calibration and params
    parser.add_argument('--train_calibrator', type=bool, default= False)
    parser.add_argument('--corners', type=str, default="/")
    parser.add_argument('--out_dir', type=str, default="./out_dir/train/calibration/")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    ccDetector = ChessboardCornerDetector()
    if args.chessboard_corner_re_detect:
        ccDetector.detect()
    if args.train_calibrator:
        chessboard_mat, gray = ccDetector.detect(args)
        calibrator = Calibrator(args)
        mtx, newmtx, dist = calibrator.fit(chessboard_mat, gray)
        undistort_img = cv2.undistort(gray, mtx, dist, newmtx)
        demo_path = "{}demo_results/undistorted_chessboard_{}.jpg".format(args.out_dir, time.ctime(time.time()))
        print(demo_path)
        cv2.imwrite(demo_path, undistort_img)
        params = {
            "mtx": newmtx,
            "dist": dist
        }
        param_path = "{}parameters/{}.pkl".format(args.out_dir, time.ctime(time.time()))
        pickle.dump(params, open(param_path, "wb"))
    print("train complete")

if __name__ == "__main__":
    main()  
