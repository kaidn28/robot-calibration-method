import numpy as np
import math
import random
def isColumn(l):
    return abs(l[3]- l[1]) > abs(l[2]- l[0])
def rowQuad(row):
    a = -(row[1]-row[3])/(row[0]-row[2])
    c = -a*row[0]-row[1]
    return a, c
def colQuad(col):
    b = -(col[2]-col[0])/(col[3]-col[1])
    c = -col[0] - b*col[1]
    return b, c
def cross(colQuad, rowQuad):
    b, c1 = colQuad
    a, c2 = rowQuad
    x = (-b*c2 + c1)/(a*b - 1)
    y = -c2 - a*x 
    return x, y
def distance(p1, p2):
    p1_ = np.array(p1)
    p2_ = np.array(p2)
    return np.linalg.norm(p1_ - p2_)
def roundToPoint5(num):
    return np.round(num + 0.5) - 0.5

def getAvgRatios(corners, num_samples):
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
def getMatBoundary(corners):
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
def initializeMat(corners, min_x, min_y, max_x, max_y, hoz_avg_ratio, ver_avg_ratio):
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
def refineMat(mat):
    print(mat[1,:])
    num_col, num_row = mat.shape[:2]
    print(num_col, num_row)
    blanks = []
    for i,col in enumerate(mat):
        for j, cell in enumerate(col):
            #print(cell)
            if np.linalg.norm(cell) == 0:
                #print(cell)
                blanks.append([i,j])
    #print(blanks)    
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
        #print(filled_neighbors)
        break 
    return mat

def getCornerMat(corners, num_samples = 5):
    hoz_avg_ratio, ver_avg_ratio = getAvgRatios(corners, num_samples)
    print(hoz_avg_ratio, ver_avg_ratio)
    min_x, min_y, max_x, max_y = getMatBoundary(corners)
    mat = initializeMat(corners, min_x, min_y, max_x, max_y, hoz_avg_ratio, ver_avg_ratio)
    mat = refineMat(mat)
    return mat, np.array([ver_avg_ratio, hoz_avg_ratio])