import cv2
from .functions import *

class ObjectDetector:
    def __init__(self, weight):
        self.weight = weight
    def train(self):
        pass
    def test(self):
        pass
    def predict(self, image):
        x = yolo(image, self.weight)
        object = []
        for i in range(len(x)):
            object.append({"class_name": x.name[i], "center": [x.xcenter[i], x.ycenter[i]]})
        return x
# a = ObjectDetector()
# print(a.predict(cv2.imread('../datasets/object_images/dataset_29072021/img_1.jpg')))
