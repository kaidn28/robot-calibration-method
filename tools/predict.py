#predict on a specific image
import pickle 
import cv2
import os
import sys
import argparse
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from regression import Regressor
from calibration import Calibrator
from object_detection import ObjectDetector
def parse_args():
    parser =argparse.ArgumentParser(description="Testing pipeline")
    parser.add_argument('--image', type=str, help='path to image', default= './datasets/object_images/dataset_29072021/img_1.jpg')
    parser.add_argument('--origin', nargs="+", help='coordinates of origin point by(x,y)', default=['583','30'])
    parser.add_argument('--out_dir', type=str, default="./out_dir")
    args = parser.parse_args()
    return args

def run_pipeline(args):
    detector = ObjectDetector()
    calibrator = Calibrator(args)
    regressor = Regressor(args)
    image = cv2.imread(args.image)
    objects = detector.predict(image)
    object_locations = []
    for o in objects:
        calib_loc = calibrator.predict(o['center'])
        reg_loc = regressor.predict(calib_loc)
        object_locations.append([o['class_name'], reg_loc])
    return object_locations
def main():
    args = parse_args()
    results = run_pipeline(args)
    print(results)

if __name__ == "__main__":
    main()