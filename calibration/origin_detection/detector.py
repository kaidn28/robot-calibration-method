#abcxyz
import sys
import os
import cv2 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from object_detection import ObjectDetector
import numpy

class OriginDetector:
    def __init__ (self):
        self.oDetector = ObjectDetector(weight="./out_dir/parameters/calibration/origin_detection/center-new.pt")
    
    def fit(self, args):
        pass

    def predict(self, img):    
        res = self.oDetector.predict(img)
        # print(res)
        #print(res)
        # print(org.xcenter[0])
        return res[0]['center']
    def test(self, args):
        pass