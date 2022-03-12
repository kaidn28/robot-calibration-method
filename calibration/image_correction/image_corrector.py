import numpy as np
import cv2 
import pickle
class ImageCorrector:
    def __init__(self):
        last_saved_params = "./out_dir/parameters/calibration/image_correction/last.pkl"
        try:
            self.params = pickle.load(open(last_saved_params, "rb"))
        except:
            print("params not found, plz train before predict")
        pass
    def fit(self, mat, img):
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = img.shape[::2]
        image_points = []
        object_points = []
        # print(mat.shape)
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

        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([object_points], [image_points], img_shape, None, None)
        if ret:
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img_shape, 1, img_shape)
            return mtx, newcameramtx, dist 
        else:
            raise Exception("Cannot calculate parameters")
    def test(self):
        pass
    def predict(self, center):
        mtx = self.params['mtx']
        dist = self.params['dist']
        inv_mtx = np.linalg.inv(mtx)
        #print("inv_mat")
        #print(inv_mtx)
        #print(mtx)
        #print(dist)
        u = center[0]
        v = center[1]
        x = u*sum(inv_mtx.T[0])
        y = v*sum(inv_mtx.T[1])
        #print(x)
        #print(y)
        r2 = x*x + y*y
        k1,k2, p1, p2, k3 = dist[0] 
        udt_x = x*(1+k1*r2+k2*r2**2+k3*r2**3) + 2*p1*x*y +p2*(r2+ 2*x**2)
        udt_y = y*(1+k1*r2+k2*r2**2+k3*r2**3) + p1*(r2+2*y**2)+2*p2*x*y
        udt_u = udt_x*sum(mtx.T[0])
        udt_v = udt_y*sum(mtx.T[1])
        return [udt_u, udt_v] 
        #print(self.params)
        return center
        # print(calib_params) 
        pass
    def undistort(self, img):
        # print(self.params)
        mtx =self.params['mtx']
        dist = self.params['dist']
        #print(dist)
        udt_img = cv2.undistort(img, mtx, dist, mtx)
        return udt_img
        #print(self.params)
