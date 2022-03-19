import numpy as np
import pickle 
import cv2
import os
import sys
import argparse
import pandas as pd
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from regression import MultiRegressor
from calibration import Calibrator
from object_detection import ObjectDetector, ObjectSegment

torch.cuda.empty_cache()
def parse_args():
    parser =argparse.ArgumentParser(description="train regression")
    
    #regresson and params
    parser.add_argument('--train_regressor', type=bool, default=True)
    parser.add_argument('--gt', type =str, default='./datasets/regression_gt/dataset_05032022.csv')
    parser.add_argument('--object_images', type=str, help='path to images', default= './datasets/object_images/dataset_05032022/')
    parser.add_argument('--object_detector', type=str, default='out_dir/parameters/object_detection/yolo_05032022.pt')
    parser.add_argument('--out_dir', type=str, default="./out_dir/")
    parser.add_argument('--calibration_mode', type = str, default="test")
    parser.add_argument('--object_detection_mode', type = str, default="test")
    parser.add_argument('--regression_mode', type = str, default="train")
    parser.add_argument('--data', type=str, default='')
    # parser.add_argument('--data', type=str, default='regression/data.csv')
    args = parser.parse_args()
    return args

def main():
    
    args = parse_args()
    # print(args)
    linear_locs = dict()
    bilinear_locs = dict()
    scaling_locs = dict()
    gt_locs = dict()


    gt_df = pd.read_csv(args.gt, index_col=0)
    # print(gt_df)
    img_center = np.array([0, 11.6])
    if not args.data:
        calibrator = Calibrator(args)
        object_detector = ObjectDetector(weight=args.object_detector)
        # object_detector = ObjectSegment()
        img_names = os.listdir(args.object_images)
        
        data = pd.DataFrame({'img_name': [], 'sca_x':[], 'sca_y': [], 'cab_x': [], 'cab_y': [], 'gt_x': [], 'gt_y': []})
        for i, n in enumerate(img_names):
            img_path = os.path.join(args.object_images, n)
            img = cv2.imread(img_path)
            # udt_img = calibrator.undistort(img)
            objects = object_detector.predict(img)
            print(n)
            # print(n)
            # print(objects)
            print("{}/{}".format(i, len(img_names)))
            for o in objects:
                # print(o['polygon'])
                # polygon = o['polygon'][0]
                # o['center'] = np.sum(polygon, axis=0)/len(polygon)
                #print(o['class_name'])
                #print(o['center'])
                # print(o['center'])
                # print('initial loc: ', o['center'])  
                try: 
                    scaling_loc, linear_loc, bilinear_loc = calibrator.transform(o['center'])
                except: 
                    continue
                gt_loc = gt_df.loc[n, ["{}_x".format(o['class_name']), "{}_y".format(o['class_name'])]].to_numpy()
                # cab_loc = np.round(cab_loc,2) 
                # scaling_loc = np.round(scaling_loc, 2)

                if not o['class_name'] in linear_locs.keys():
                    linear_locs[o['class_name']] = np.empty((0,2))
                linear_locs[o['class_name']] = np.concatenate((linear_locs[o['class_name']], linear_loc.reshape(1,2)))

                if not o['class_name'] in bilinear_locs.keys():
                    bilinear_locs[o['class_name']] = np.empty((0,2))
                bilinear_locs[o['class_name']] = np.concatenate((bilinear_locs[o['class_name']], bilinear_loc.reshape(1,2)))
                
                if not o['class_name'] in scaling_locs.keys():
                    scaling_locs[o['class_name']] = np.empty((0,2))
                scaling_locs[o['class_name']] = np.concatenate((scaling_locs[o['class_name']], scaling_loc.reshape(1,2)))

                if not o['class_name'] in gt_locs.keys():
                    gt_locs[o['class_name']] = np.empty((0,2))
                gt_locs[o['class_name']] = np.concatenate((gt_locs[o['class_name']], gt_loc.reshape(1,2)))
                data = data.append({
                        'img_name': n,
                        'class_name': o['class_name'],
                        'sca_x': scaling_loc[0],
                        'sca_y': scaling_loc[1],
                        'linear_x': linear_loc[0],
                        'linear_y': linear_loc[1],
                        'bilinear_x': bilinear_loc[0],
                        'bilinear_y': bilinear_loc[1],
                        'gt_x': gt_loc[0],
                        'gt_y': gt_loc[1]
                        }, ignore_index=True)
                
                if o['class_name'] == 'silver' and n == '99.jpg':
                    img_cp = img.copy()
                    img_cp2 = img.copy()
                    center = (np.int(o['center'][0]), np.int(o['center'][1]))

                    point = np.round(linear_loc/2.9,2)
                    # for i in polygon:
                    #     # print(i)
                    #     point = (int(i[0]), int(i[1]))
                    #     print(point)
                    #     cv2.circle(img_cp, point, 5, (0,0,255), -1)    
                    cv2.circle(img_cp, center, 5, (0,0,255), -1)
                    cv2.putText(img_cp, f"{list(point)}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 1, cv2.LINE_AA,False)

                    
                    cv2.imshow(n, img_cp)
                    cv2.imshow(n+n, img_cp2)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
            

        # data['sca_err_x'] = np.abs(data['sca_x'] - data['gt_x'])
        # data['sca_err_y'] = np.abs(data['sca_y'] - data['gt_y'])
        # data['sca_err'] = np.round(np.sqrt(data['sca_err_x']**2 + data['sca_err_y']**2), 2)
        # data['cab_err_x'] = np.round(np.abs(data['cab_x'] - data['gt_x']),2)
        # data['cab_err_y'] = np.round(np.abs(data['cab_y'] - data['gt_y']),2)
        # data['cab_err'] = np.round(np.sqrt(data['cab_err_x']**2 + data['cab_err_y']**2), 2)
        # # print(cab_locs['red'].shape)
        # # print(gt_locs['red'].shape)
        data.to_csv('regression/data.csv', index=False)

    else: 
        df = pd.read_csv(args.data)
        for object_name in df['class_name'].unique():
            # print(object_name)
            # print(type(object_name))
            linear_locs[object_name] = np.empty((0,2))
            bilinear_locs[object_name] = np.empty((0,2))
            scaling_locs[object_name] = np.empty((0,2))
            gt_locs[object_name] = np.empty((0,2))

        for i in range(df.shape[0]):
            rec = df.loc[i, ['class_name', 'sca_x', 'sca_y', 'linear_x', 'linear_y','bilinear_x', 'bilinear_y', 'gt_x', 'gt_y']]
            # print(rec)
            linear_loc = rec[['linear_x', 'linear_y']].to_numpy().reshape(1,2)
            bilinear_loc = rec[['bilinear_x', 'bilinear_y']].to_numpy.reshape(1,2)
            sca_loc = rec[['sca_x', 'sca_y']].to_numpy().reshape(1,2)
            gt_loc = rec[['gt_x', 'gt_y']].to_numpy().reshape(1,2)
            # print(rec['class_name'])

            # print(np.concatenate([scaling_locs[rec['class_name']], sca_loc]))
            linear_locs[rec['class_name']] = np.concatenate([linear_locs[rec['class_name']], linear_loc])
            linear_locs[rec['class_name']] = np.concatenate([bilinear_locs[rec['class_name']], bilinear_loc])
            scaling_locs[rec['class_name']] = np.concatenate([scaling_locs[rec['class_name']], sca_loc])
            gt_locs[rec['class_name']] = np.concatenate([gt_locs[rec['class_name']], gt_loc])

    
    regressor = MultiRegressor(args)
    regressor.fit(linear_locs, gt_locs, img_center)



        # print(n)
        # print(gt_df.loc[n].to_numpy())

        
        
        


    print("train complete")

if __name__ == "__main__":
    main()  
