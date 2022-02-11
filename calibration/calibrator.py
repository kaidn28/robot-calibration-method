import numpy as np
import cv2 
class Calibrator:
    def __init__(self, args):
        pass
    def fit(self, mat, gray):
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
            print(mtx)
            print(dist)
            print(rvecs)
            print(tvecs)
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (gray_w,gray_h), 1, (gray_w, gray_h))
            # undistort
            dst = cv2.undistort(gray, mtx, dist, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            cv2.imwrite("undistorted.jpg", dst)
        pass
    def test(self):
        pass
    def predict(self): 
        pass
