import pandas as pd
import numpy as np
import os 
import cv2
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
class Regressor:
    def __init__(self, args):
        self.data = pd.read_csv(args.data).loc[:, ['x', 'y', 'x_gt', 'y_gt']]
        self.data.info()
        drawingBoard = np.zeros((700, 700)) + 255
        drawingBoard = cv2.merge([drawingBoard, drawingBoard, drawingBoard])
        print(self.data)
        data_draw = self.data.applymap(lambda x: int(x)*10 + 300).to_numpy()
        # print(data_draw)
        for x1, y1, x2, y2 in data_draw:
            cv2.line(drawingBoard, (x1,y1), (x2,y2), (0, 255, 0), 2)
            cv2.circle(drawingBoard, (x1,y1), 2, (0,0,255), -1)
            cv2.circle(drawingBoard, (x2,y2), 2, (255,0,0), -1)
        #cv2.imshow('abc', drawingBoard)
        
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # cv2.imwrite('visualize.jpg', drawingBoard)
        # print(self.data)
        self.drawingBoard = drawingBoard
        #print(data_proc)