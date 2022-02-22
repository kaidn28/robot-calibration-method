import numpy as np
import cv2 
import pickle
class Calibrator:
    def __init__(self, args):
        pass
    def fit(self, mat, img):
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = img
        gray_w, gray_h = gray.shape[::-1]
        image_points = []
        object_points = []
        print(mat.shape)
        for col in range(mat.shape[0]):
            for row in range(mat.shape[1]):
                cell = mat[col, row]
                if cell[0] != 0:
                    cell_format = np.float32(cell)
                    cell_idx_format = np.float32(np.array([col, row,0]))
                    image_points.append(cell_format.reshape(-1, 2))
                    object_points.append(cell_idx_format.reshape(-1, 3))
        
        image_points = np.array(image_points)
        object_points = np.array(object_points)

        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([object_points], [image_points], (gray_w, gray_h), None, None)
        if ret:
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (gray_w,gray_h), 1, (gray_w, gray_h))
            return mtx, newcameramtx, dist 
        else:
            raise Exception("Cannot calculate parameters")
    def test(self):
        pass
    def predict(self, center):
        last_saved_params = "./out_dir/train/calibration/parameters/last.pkl"
        calib_params = pickle.load(open(last_saved_params, "rb"))
        print(calib_params) 
        pass
    def undistort(self, img):
        calib_params = pickle.load()
