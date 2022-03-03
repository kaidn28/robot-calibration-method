import cv2
import numpy as np
from .functions import *

class ObjectDetector:
    def __init__(self, weight = "./out_dir/parameters/object_detection/super-best.pt"):
        self.weight = weight
    def train(self):
        pass
    def test(self):
        pass
    def predict(self, image):
        x = yolo(image, self.weight)
        objects = []
        for i in range(len(x)):
            objects.append({"class_name": x.name[i], "center": np.array([x.xcenter[i], x.ycenter[i]])})
        return objects
# a = ObjectDetector()
# print(a.predict(cv2.imread('../datasets/object_images/dataset_29072021/img_1.jpg')))
