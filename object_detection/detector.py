import cv2
import numpy as np
from .functions import *

class ObjectDetector:
    def __init__(self, weight = "./out_dir/parameters/object_detection/super-best.pt"):
        self.weight = weight
        self.model = yolo(self.weight)
    def train(self):
        pass
    def test(self):
        pass
    def predict(self, image):
        
        img = image[..., ::-1]
        results = self.model(img)
        a = results.pandas().xywh[0]
        objects = []
        for i in range(len(a)):
            objects.append({"class_name": a.name[i], "center": np.array([a.xcenter[i], a.ycenter[i]])})
        return objects
# a = ObjectDetector()
# print(a.predict(cv2.imread('../datasets/object_images/dataset_29072021/img_1.jpg')))
