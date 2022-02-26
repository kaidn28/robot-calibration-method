#abcxyz
import sys
import os
import cv2 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from object_detection import ObjectDetector
import numpy

def origin(img):
    a = ObjectDetector("./out_dir/train/detection/center.pt")
    res = a.predict(img)
    org = res[res["name"]=="x"]
    # print(org.xcenter[0])
    return numpy.array([org.xcenter[0], org.ycenter[0]])