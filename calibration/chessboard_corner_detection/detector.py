import cv2
import pickle
from .functions import *
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
class ChessboardCornerDetector:
    #calculate average pixel/cm based on x and y axis
    def fit():
        pass
    


    def detect(self, img):
        self.img = img
        if len(self.img.shape) == 3:
            img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else: 
            img_gray = self.img.copy()

        dst = cv2.Canny(img_gray, 50, 200, None, 3)
        # Copy edges to the images that will display the results in BGR
        cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

        linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 80, 50)    
        #print(linesP)
        columns = []
        rows = []
        if linesP is not None:
            for i in linesP:
                l = i[0]
                if isColumn(l): 
                    columns.append(l)
                    cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
                else:
                    rows.append(l)
                    cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
        #print(len(rows))
        #print(len(columns))
        corners = []
        for r in rows:
            for c in columns:
                b,c1 = colQuad(c)
                a, c2 = rowQuad(r)
                if (r[0] + b*r[1] + c1)*(r[2]+b*r[3] +c1) < 0 and (a*c[0] + c[1] + c2)*(a*c[2]+c[3] +c2) < 0:  
                    x, y = cross((b,c1), (a,c2))
                    checked = False 
                    for i, p in enumerate(corners):
                        if distance(p, (x,y))< 10:
                            #print(distance(p, (x,y)))
                            corners[i] = ((x+ p[0])/2, (y+p[1])/2)
                            checked = True
                            break
                    if not checked:        
                        corners.append(np.array([x,y]))
        #print(corners)
        mat, ratios = getCornerMat(corners)
        # print(mat.shape)
        # for i, (x, y) in enumerate(corners):
        #     cv2.circle(self.img, (int(x),int(y)), 5, (0,0, 255), -1)
        #     #cv2.putText(self.img, str(i), (int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        # cv2.imshow("abc", self.img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # corner_mat_path = "./out_dir/train/calibration/corner_detection/corner_mat_{}.pkl".format(time.ctime(time.time()))
        # last_corner_mat_path = "./out_dir/train/calibration/corner_detection/corner_mat_last.pkl"
        # pickle.dump(mat, open(corner_mat_path, "wb"))
        # pickle.dump(mat, open(last_corner_mat_path, "wb"))
        return mat, ratios
    

        