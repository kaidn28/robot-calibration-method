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
    def getCornerMat(self, corners, num_samples = 5):
        random.seed()
        hoz_mins = []
        ver_mins = []
        for i in range(num_samples):
            id = random.randrange(0, len(corners))
            #print(id)
            hoz_min = 9999
            ver_min = 9999
            for j,c in enumerate(corners):
                dis = distance(c, corners[id])
                hoz = np.abs(c[0] - corners[id][0])
                ver = np.abs(c[1] - corners[id][1])
                if hoz <hoz_min and hoz > 10:
                    hoz_min = hoz
                if ver < ver_min and ver > 10:
                    ver_min = ver
            hoz_mins.append(hoz_min)
            ver_mins.append(ver_min) 
        hoz_mins.sort()
        ver_mins.sort()
        hoz_avg_ratio = hoz_mins[num_samples//2]
        ver_avg_ratio = ver_mins[num_samples//2]
        min_x = 99999
        max_x = 0
        min_y = 99999
        max_y = 0
        for i, (x,y) in enumerate (corners):
            if x > max_x:
                max_x = x
            elif x < min_x: 
                min_x = x
            if y > max_y:
                max_y = y
            elif y < min_y:
                min_y = y
        print(min_x, min_y, max_x, max_y)
        num_cols = math.ceil((max_x-min_x)/hoz_avg_ratio)
        num_rows = math.ceil((max_y-min_y)/ver_avg_ratio)
        print(num_cols)
        print(num_rows)
    def detect(self, args):
        self.img_path = args.chessboard_image
        try: 
            self.img = cv2.imread(self.img_path)
        except:
            raise Exception('Image not found.')

        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
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
        print(corners)
        mat = self.getCornerMat(corners)
        for i, (x, y) in enumerate(corners):
            cv2.circle(self.img, (int(x),int(y)), 5, (0,0, 255), -1)
            #cv2.putText(self.img, str(i), (int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow("abc", self.img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    

        