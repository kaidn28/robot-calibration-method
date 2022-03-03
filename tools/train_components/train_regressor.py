import numpy as np
import pickle 
import cv2
import os
import sys
import argparse
import pandas as pd
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from regression import Regressor, regressor
from calibration import Calibrator
from object_detection import ObjectDetector

torch.cuda.empty_cache()
def parse_args():
    parser =argparse.ArgumentParser(description="train regression")
    
    #regresson and params
    parser.add_argument('--train_regressor', type=bool, default=True)
    parser.add_argument('--gt', type =str, default='./datasets/regression_gt/dataset_29072021.csv')
    parser.add_argument('--object_images', type=str, help='path to images', default= './datasets/object_images/dataset_29072021/')
    parser.add_argument('--out_dir', type=str, default="./out_dir/")
    parser.add_argument('--calibration_mode', type = str, default="test")
    parser.add_argument('--object_detection_mode', type = str, default="test")
    parser.add_argument('--regression_mode', type = str, default="train")
    args = parser.parse_args()
    return args

def main():
    
    args = parse_args()
    print(args)
    
    calibrator = Calibrator(args)
    object_detector = ObjectDetector()
    regressor = Regressor(args)


    gt_df = pd.read_csv(args.gt, index_col=0)
    print(gt_df)
    img_names = os.listdir(args.object_images)
    
    cab_locs = dict()
    gt_locs = dict()
    for n in img_names:
        img_path = os.path.join(args.object_images, n)
        img = cv2.imread(img_path)
        udt_img = calibrator.undistort(img)
        objects = object_detector.predict(udt_img)
        print(n)
        for o in objects:
            cab_loc = calibrator.transform(o['center'])
            #print(o['class_name'])
            #print(cab_loc)    
            gt_loc = gt_df.loc[n, ["{}_x".format(o['class_name']), "{}_y".format(o['class_name'])]].to_numpy()
            #print(gt_loc)

            if not o['class_name'] in cab_locs.keys():
                cab_locs[o['class_name']] = np.empty((0,2))
            cab_locs[o['class_name']] = np.concatenate((cab_locs[o['class_name']], cab_loc.reshape(1,2)))

            if not o['class_name'] in gt_locs.keys():
                gt_locs[o['class_name']] = np.empty((0,2))
            gt_locs[o['class_name']] = np.concatenate((gt_locs[o['class_name']], gt_loc.reshape(1,2)))
    print(cab_locs['red'].shape)
    print(gt_locs['red'].shape)

    regressor.fit(cab_locs, gt_locs)



        # print(n)
        # print(gt_df.loc[n].to_numpy())

        
        
        


    print("train complete")

if __name__ == "__main__":
    main()  
