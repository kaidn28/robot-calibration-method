import cv2
from .functions import *

class ObjectDetector:
    def __init__(self):
        pass
    def train(self):
        pass
    def test(self):
        pass
    def predict(self, image):
        x = yolo(image)
        object = []
        for i in range(len(x)):
            object.append({"class_name": x.name[i], "center": [x.xcenter[i], x.ycenter[i]]})
        return object
# a = ObjectDetector()
# print(a.predict(cv2.imread('../datasets/object_images/dataset_29072021/img_1.jpg')))
