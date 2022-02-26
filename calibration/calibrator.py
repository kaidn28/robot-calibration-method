from .chessboard_corner_detection import ChessboardCornerDetector
from .image_correction import ImageCorrector
from .origin_detection import OriginDetector
import cv2
import numpy as np
import pickle
import time
class Calibrator:
    def __init__(self):
        self.ccDetector = ChessboardCornerDetector()
        self.imCorrector = ImageCorrector()
        self.orDetector = OriginDetector()
    def fit(self, args):
        try:
            img = cv2.imread(args.chessboard_image)
        except: 
            print("chessboard image not found!")

        chessboard_mat, gray = self.ccDetector.detect(img)

        mtx, newmtx, dist = self.imCorrector.fit(chessboard_mat, gray)
        undistort_img = cv2.undistort(img, mtx, dist, newmtx)
        demo_path = "{}image_correction/demo_results/undistorted_chessboard_{}.jpg".format(args.out_dir, time.ctime(time.time()))
    
        print(demo_path)
        cv2.imwrite(demo_path, undistort_img)
    
        #re-detect after distortion
        self.ccDetector.detect(undistort_img)
        params = {
            "newmtx": newmtx,
            "mtx": mtx,
            "dist": dist
        }
        origin = self.orDetetector.predict(undistort_img)


        #params in chessboard detector
        corner_mat_path ="{}corner_detection/corner_mat_{}.pkl".format(args.out_dir, time.ctime(time.time()))
        last_corner_mat_path = "{}corner_detection/corner_mat_last.pkl".format(args.out_dir)

        #params in image correction
        correction_param_path = "{}image_correction/parameters/{}.pkl".format(args.out_dir, time.ctime(time.time()))
        last_correction_param_path = "{}image_correction/parameters/last.pkl".format(args.out_dir)
        
        #params in origin detection
        origin_path = "{}origin_detection/origin/{}.pkl".format(args.out_dir, time.ctime(time.time()))
        last_origin_path = "{}origin_detection/origin/last.pkl".format(args.out_dir)
        
        pickle.dump(chessboard_mat, open(corner_mat_path, "wb"))
        pickle.dump(chessboard_mat, open(last_corner_mat_path, "wb"))
        pickle.dump(params, open(correction_param_path, "wb"))
        pickle.dump(params, open(last_correction_param_path, "wb"))
        pickle.dump(origin, open(origin_path, "wb"))
        pickle.dump(origin, open(last_origin_path, "wb"))
        

        print("train complete")
        pass
    def predict(self, args):
        pass
    def test(self, args):
        pass
