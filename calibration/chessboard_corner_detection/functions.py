import numpy as np

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