import cv2
import numpy as np
from .functions import *
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from imantics import Polygons, Mask
import os
class ObjectDetector:
    def __init__(self, config = None, weight = "./out_dir/parameters/object_detection/super-best.pt"):
        self.weight = weight
        self.model = yolo(self.weight)
        self.config = config
        if cfg is None:
            self.config = cfg()
        

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

    def maskRCNN(self, img):
        predictor = DefaultPredictor(self.config)

        outputs = predictor(img)
        # print(outputs['instances'].pred_masks[0])
        masks = outputs['instances'].pred_masks
        masks = masks.int()
        o = []
        for i, mask in enumerate(masks):
            polygons = Mask(np.array(masks[0])).polygons()
            print(outputs['instances'].pred_classes.tolist()[i] == 0)
            if outputs['instances'].pred_classes.tolist()[i] == 0:
                o.append({"class_name": "black", "polygon": polygons.points})
            elif outputs['instances'].pred_classes.tolist()[i] == 1:
                o.append({"class_name": "white", "polygon": polygons.points})
            elif outputs['instances'].pred_classes.tolist()[i] == 2:
                o.append({"class_name": "silver", "polygon": polygons.points})

        return o 
# a = ObjectDetector()
    
# a = ObjectDetector()
# print(a.predict(cv2.imread('../datasets/object_images/dataset_29072021/img_1.jpg')))
