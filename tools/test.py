#test the accuracy of the algorithm on a list of images
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
    parser.add_argument('--images', type=str, help='path to images', default= './datasets/object_images/dataset_29072021/')
    parser.add_argument('--origin', nargs="+", help='coordinates of origin point by(x,y)', default=['583','30'])
    parser.add_argument('--out_dir', type=str, default="./out_dir")
    args = parser.parse_args()
    return args

def run_pipeline(args):
    detector = ObjectDetector()
    calibrator = Calibrator(args)
    regressor = Regressor(args)
    image_names = os.listdir(args.images)
    results = []
    for n in image_names:
        image_path = os.path.join(args.images, n)
        print(image_path)
        image = cv2.imread(image_path)
        objects = detector.predict(image)
        object_locations = []
        for o in objects:
            print(o['center'])
            calib_loc = calibrator.predict(o['center'])
            reg_loc = regressor.predict(calib_loc)
            object_locations.append(reg_loc)
        results.append([n, object_locations])
    return results
def main():
    args = parse_args()
    results = run_pipeline(args)
    print(results)

if __name__ == "__main__":
    main()