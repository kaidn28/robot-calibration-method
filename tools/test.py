#test the accuracy of the algorithm on a list of images
import pickle 
import cv2
import os
import sys
import argparse
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from regression import MultiRegressor
from calibration import Calibrator
from object_detection import ObjectDetector
def parse_args():
    parser =argparse.ArgumentParser(description="Testing pipeline")
    parser.add_argument('--images', type=str, help='path to images', default= './datasets/object_images/dataset_29072021/')
    parser.add_argument('--origin', nargs="+", help='coordinates of origin point by(x,y)', default=['583','30'])
    parser.add_argument('--out_dir', type=str, default="./out_dir/")
    parser.add_argument('--object_detection_weight', type = str, help="path to object detection model params", default='./out_dir/parameters/object_detection/super-best.pt')
    parser.add_argument('--calibration_mode', type = str, default = "test")
    parser.add_argument('--object_detection_mode', type = str, default = "test")
    parser.add_argument('--regression_mode', type = str, default="test")
    parser.add_argument('--classes', type =list, default=["red", "yellow", "green"])
    args = parser.parse_args()
    return args

def run_pipeline(args):
    detector = ObjectDetector(args.object_detection_weight)
    calibrator = Calibrator(args)
    regressor = MultiRegressor(args)
    image_names = os.listdir(args.images)
    results = []
    # for n in image_names:
    #     image_path = os.path.join(args.images, n)
    #     print(image_path)
    #     image = cv2.imread(image_path) 
    #     objects = detector.predict(image)
    #     object_locations = []
    #     for o in objects:
    #         print(o['center'])
    #         calib_loc = calibrator.predict(o['center'])
    #         # reg_loc = regressor.predict(calib_loc)
    #         # object_locations.append(reg_loc)
    #         print(calib_loc)
            
    #     results.append([n, object_locations])
    # n = image_names[0]
    # image_path = os.path.join(args.images, n)
    # print(image_path)
    # image = cv2.imread(image_path) 
    # objects = detector.predict(image)
    # object_locations = []
    # print("detect first")
    # for o in objects:
    #     #print(o['center'])
    #     calib_loc = calibrator.predict(o['center'])
    #     # reg_loc = regressor.predict(calib_loc)
    #     # object_locations.append(reg_loc)
        
    #     print(calib_loc)
    for n in image_names:        
        image_path = os.path.join(args.images, n)
        print(n)
        image = cv2.imread(image_path) 
        udt_image = calibrator.undistort(image)
        objects = detector.predict(udt_image)
        #print("calib first")
        #print(type(objects))
        for o in objects:
            #print(o['center'])
            
            cab_loc = calibrator.transform(o['center'])
            #print(o['center'])
            #print(cab_loc)
            reg_loc= regressor.predict(o['class_name'], cab_loc)
            results.append([o["class_name"], reg_loc])

            
        #print(results)
def main():
    args = parse_args()
    results = run_pipeline(args)
    print(results)

if __name__ == "__main__":
    main()