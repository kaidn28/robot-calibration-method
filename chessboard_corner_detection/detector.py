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
    def train():
        pass
    
    def getAvgRatios(self, corners, num_samples):
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
        return hoz_avg_ratio, ver_avg_ratio
    # calculate the smallest and largest coordinates on x,y axis of corners
    def getMatBoundary(self, corners):
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
        return min_x, min_y, max_x, max_y
    # initialize corner matrix with detected corners
    def initializeMat(self, corners, min_x, min_y, max_x, max_y, hoz_avg_ratio, ver_avg_ratio):
        num_cols = math.ceil((max_x-min_x)/hoz_avg_ratio)
        num_rows = math.ceil((max_y-min_y)/ver_avg_ratio)
        # matx = np.zeros((num_cols, num_rows))
        # maty = np.zeros((num_cols, num_rows))
        mat = np.zeros((num_cols, num_rows, 2))
        #print(mat)
        for i, xy in enumerate(corners):
            col = math.floor((xy[0]-min_x)*(num_cols-1)/(max_x - min_x) + 0.5)
            row = math.floor((xy[1]-min_y)*(num_rows-1)/(max_y - min_y)+ 0.5)
            #print(xy)
            #print(col, row)
            # matx[col, row] = xy[0]
            # maty[col, row] = xy[1]
            mat[col, row] = xy
        #print(mat[:10, :10]) 
        # np.savetxt("./chessboard_corner_detection/mat_x.csv", matx, delimiter=",")
        # np.savetxt("./chessboard_corner_detection/mat_y.csv", maty, delimiter=",")
        return mat
    #fill in the blanks in the initialized matrix
    def refineMat(self, mat):
        num_col, num_row = mat.shape[:2]
        print(num_col, num_row)
        blanks = []
        for i,col in enumerate(mat):
            for j, cell in enumerate(col):
                print(cell)
                if np.linalg.norm(cell) == 0:
                    print(cell)
                    blanks.append([i,j])
        print(blanks)    
        while blanks:
            blank = blanks[0]
            col_blank = blank[0]
            row_blank = blank[1]
            neighbors = [
                (col_blank-1, row_blank-1),
                (col_blank-1, row_blank), 
                (col_blank-1, row_blank+1),
                (col_blank, row_blank-1),
                (col_blank, row_blank+1),
                (col_blank+1, row_blank-1),
                (col_blank+1, row_blank),
                (col_blank+1, row_blank+1)
                ]
            filled_neighbors = []
            for n in neighbors:
                if 0<= n[0] < num_col and 0<= n[1] < num_row:
                    filled_neighbors.append((n, mat[n[0], n[1]]))
            print(filled_neighbors)
            break 
    def getCornerMat(self, corners, num_samples = 5):
        hoz_avg_ratio, ver_avg_ratio = self.getAvgRatios(corners, num_samples)
        print(hoz_avg_ratio, ver_avg_ratio)
        min_x, min_y, max_x, max_y = self.getMatBoundary(corners)
        mat = self.initializeMat(corners, min_x, min_y, max_x, max_y, hoz_avg_ratio, ver_avg_ratio)
        mat = self.refineMat(mat)
        return mat
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
    
    

        