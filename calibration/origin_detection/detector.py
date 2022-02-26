#abcxyz
import sys
import os
import cv2 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from object_detection import ObjectDetector
import numpy

class OriginDetector:
    def __init__ (self):
        self.oDetector = ObjectDetector("./out_dir/train/detection/center.pt")
    
    def fit(self, args):
        pass

    def predict(self, img):    
        res = self.oDetector.predict(img)
        org = res[res["name"]=="x"]
        # print(org.xcenter[0])
        return numpy.array([org.xcenter[0], org.ycenter[0]])
    def test(self, args):
        pass