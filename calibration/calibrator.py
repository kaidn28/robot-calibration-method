from .chessboard_corner_detection import ChessboardCornerDetector
from .image_correction import ImageCorrector
from .origin_detection import OriginDetector
from .interpolation import Interpolator
import cv2
import numpy as np
import pickle
import time
class Calibrator:
    def __init__(self, args):
        self.ccDetector = ChessboardCornerDetector()
        self.imCorrector = ImageCorrector()
        self.orDetector = OriginDetector()
        self.interpolator = Interpolator()
        
        out_dir = args.out_dir
        params_dir = out_dir + 'parameters/calibration/'
        demo_dir = out_dir + 'demo_results/calibration/'
        
        #params path
        self.chessboard_detection_params_path = params_dir + 'corner_detection/'
        self.image_correction_params_path = params_dir + 'image_correction/'
        self.origin_detection_params_path = params_dir + 'origin_detection/'
        
        
        #demo path
        self.chessboard_detection_demo_path = demo_dir + 'image_correction/'
        self.image_correction_demo_path = demo_dir + 'image_correction/'
        self.origin_detection_demo_path = demo_dir + 'origin_detection/'

        if args.calibration_mode == "test":
            self.chessboard_mat = pickle.load(open(self.chessboard_detection_params_path + "last.pkl", "rb"))
            self.correction_params = pickle.load(open(self.image_correction_params_path + "last.pkl", "rb"))
            self.transformation_params = pickle.load(open(self.origin_detection_params_path + "last.pt", "rb"))
            print(self.transformation_params)
        elif args.calibration_mode == "train":
            self.cell_length = args.cell_length
    def fit(self, img):

        chessboard_mat, img_cell_lengths = self.ccDetector.detect(img)
        # print(chessboard_mat.shape)
        # corners = []
        # for i in range(chessboard_mat.shape[0]):
        #     for j in range(chessboard_mat.shape[1]):
        #         if np.linalg.norm(chessboard_mat[i,j] >0):
        #             x, y = chessboard_mat[i,j]
        #             print(x, y, "\t", i, j)
        #             x_int, y_int = int(x), int(y)
        #             # corners.append(np.array([x_int, y_int]))
        #             cv2.circle(img, (x_int, y_int), 5, (255,0,0), -1)
        mtx, newmtx, dist = self.imCorrector.fit(chessboard_mat, img)

        undistort_img = cv2.undistort(img, mtx, dist, newmtx)
        # cv2.imshow("xyz", img)
        # cv2.imshow('abc', undistort_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    
        #re-detect after distortion
        # chessboard_mat, img_cell_lengths = self.ccDetector.detect(undistort_img)
        # print(chessboard_mat.shape)
        corners = []
        # for i in range(chessboard_mat.shape[0]):
        #     for j in range(chessboard_mat.shape[1]):
        #         if np.linalg.norm(chessboard_mat[i,j] >0):
        #             x, y = chessboard_mat[i,j]
        #             print(x, y, "\t", i, j)
        #             x_int, y_int = int(x), int(y)
        #             # corners.append(np.array([x_int, y_int]))
        #             cv2.circle(undistort_img, (x_int, y_int), 5, (255,0,0), -1)
        # cv2.imshow("xyz", img)
        # cv2.imshow('abc', undistort_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        correction_params = {
            "newmtx": newmtx,
            "mtx": mtx,
            "dist": dist
        }
        # origin = self.orDetector.predict(undistort_img)
        origin = self.orDetector.predict(img)
        transformation_params = {
            "origin": origin,
            "cell_length": self.cell_length,
            "img_cell_lengths": img_cell_lengths
        }


        #params in chessboard detector
        corner_mat_path ="{}corner_mat_{}.pkl".format(self.chessboard_detection_params_path, time.ctime(time.time()))
        last_corner_mat_path = "{}last.pkl".format(self.chessboard_detection_params_path)

        #params in image correction
        correction_param_path = "{}{}.pkl".format(self.image_correction_params_path, time.ctime(time.time()))
        last_correction_param_path = "{}last.pkl".format(self.image_correction_params_path)
        
        #params in origin detection
        origin_path = "{}{}.pt".format(self.origin_detection_params_path, time.ctime(time.time()))
        last_origin_path = "{}last.pt".format(self.origin_detection_params_path)
        
        #save params
        pickle.dump(chessboard_mat, open(corner_mat_path, "wb"))
        pickle.dump(chessboard_mat, open(last_corner_mat_path, "wb"))
        pickle.dump(correction_params, open(correction_param_path, "wb"))
        pickle.dump(correction_params, open(last_correction_param_path, "wb"))
        pickle.dump(transformation_params, open(origin_path, "wb"))
        pickle.dump(transformation_params, open(last_origin_path, "wb"))
        
        #save demo images

        cd_demo_path =  "{}undistorted_chessboard_{}.jpg".format(self.image_correction_demo_path, time.ctime(time.time()))

        cv2.imwrite(cd_demo_path, undistort_img)
        # missing chessboard detection and origin detection demo


        print("train complete")
        pass
    def undistort(self, img):
        mtx = self.correction_params['mtx']
        dist = self.correction_params['dist']
        newmtx = self.correction_params['newmtx']
        #mtx, dist, newmtx = self.correction_params['mtx', 'dist', 'newmtx']
        undistort_img = cv2.undistort(img, mtx, dist, newmtx)
        return undistort_img
        
    def transform(self, point):
        object_loc = point 
        origin_loc = self.transformation_params['origin']
        corner_mat = self.chessboard_mat
        #print(object_loc)
        #print(origin_loc)
        cell_length = self.transformation_params['cell_length']
        
        img_cell_lengths = self.transformation_params['img_cell_lengths']
        #print(img_cell_lengths)
        # print(f"img_cell_length: {img_cell_lengths}")
        
        return self.interpolator.predict(object_loc, origin_loc, img_cell_lengths, cell_length, corner_mat)

    def test(self, args):
        pass
