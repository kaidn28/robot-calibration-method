import numpy as np
import pickle 
import cv2
import os
import sys
import argparse
import pandas as pd
from object_detection.segment import ObjectSegment
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from regression import MultiRegressor
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
    # print(args)
    
    calibrator = Calibrator(args)
    object_detector = ObjectSegment()
    


    gt_df = pd.read_csv(args.gt, index_col=0)
    # print(gt_df)
    img_names = os.listdir(args.object_images)
    
    cab_locs = dict()
    gt_locs = dict()
    scaling_locs = dict()
    data = pd.DataFrame({'img_name': [], 'sca_x':[], 'sca_y': [], 'cab_x': [], 'cab_y': [], 'gt_x': [], 'gt_y': []})
    for n in img_names:
        img_path = os.path.join(args.object_images, n)
        img = cv2.imread(img_path)
        _, img_center = calibrator.transform((img.shape[1]/2, img.shape[0]/2))
        udt_img = calibrator.undistort(img)
        objects = object_detector.predict(udt_img)
        # print(n)
        # print(n)
        for o in objects:
            if o['class_name'] == 'red':
                print(o)
                # print('initial loc: ', o['center'])  
                cab_loc, scaling_loc = calibrator.transform(o['center'])
                gt_loc = gt_df.loc[n, ["{}_x".format(o['class_name']), "{}_y".format(o['class_name'])]].to_numpy()
                cab_loc = np.round(cab_loc,2)
                scaling_loc = np.round(scaling_loc, 2)

                if not o['class_name'] in cab_locs.keys():
                    cab_locs[o['class_name']] = np.empty((0,2))
                cab_locs[o['class_name']] = np.concatenate((cab_locs[o['class_name']], cab_loc.reshape(1,2)))
                
                if not o['class_name'] in scaling_locs.keys():
                    scaling_locs[o['class_name']] = np.empty((0,2))
                scaling_locs[o['class_name']] = np.concatenate((scaling_locs[o['class_name']], scaling_loc.reshape(1,2)))

                if not o['class_name'] in gt_locs.keys():
                    gt_locs[o['class_name']] = np.empty((0,2))
                gt_locs[o['class_name']] = np.concatenate((gt_locs[o['class_name']], gt_loc.reshape(1,2)))
                data = data.append({
                        'img_name': n,
                        'sca_x': scaling_loc[0],
                        'sca_y': scaling_loc[1],
                        'cab_x': cab_loc[0],
                        'cab_y': cab_loc[1],
                        'gt_x': gt_loc[0],
                        'gt_y': gt_loc[1]
                        }, ignore_index=True)
                
                # img_cp = udt_img.copy()
                # center = (np.int(o['center'][0]), np.int(o['center'][1]))

                # cv2.circle(img_cp, center, 5, (0,0,255), -1)
                # cv2.putText(img_cp, f"{list(cab_loc)}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 1, cv2.LINE_AA,False)
                # for i, c in enumerate(ne_co):
                #     corner = (int(c[0]), int(c[1]))
                #     corner_real = (ne_cor_real[i][0], ne_cor_real[i][1])
                #     cv2.circle(img_cp, corner, 2, (255,0,0), -1)
                #     cv2.putText(img_cp, f"{list(corner_real)}", corner, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1, cv2.LINE_AA,False)
                    
                # cv2.imshow('abc', img_cp)
                # cv2.waitKey()
                # cv2.destroyAllWindows()

            else:
                continue
        

    # data['sca_err_x'] = np.abs(data['sca_x'] - data['gt_x'])
    # data['sca_err_y'] = np.abs(data['sca_y'] - data['gt_y'])
    # data['sca_err'] = np.round(np.sqrt(data['sca_err_x']**2 + data['sca_err_y']**2), 2)
    # data['cab_err_x'] = np.round(np.abs(data['cab_x'] - data['gt_x']),2)
    # data['cab_err_y'] = np.round(np.abs(data['cab_y'] - data['gt_y']),2)
    # data['cab_err'] = np.round(np.sqrt(data['cab_err_x']**2 + data['cab_err_y']**2), 2)
    # # print(cab_locs['red'].shape)
    # # print(gt_locs['red'].shape)
    # data.to_csv('data.csv', index=False)
    regressor = MultiRegressor(args)
    regressor.fit(scaling_locs, gt_locs, img_center)



        # print(n)
        # print(gt_df.loc[n].to_numpy())

        
        
        


    print("train complete")

if __name__ == "__main__":
    main()  
