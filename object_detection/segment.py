import cv2
import numpy as np
from .functions import *
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from imantics import Polygons, Mask
import os


class ObjectSegment:
    def __init__(self, cfg):
        self.cfg = cfg

    def train(self):
        pass

    def test(self):
        pass

    def predict(self, img):
        predictor = DefaultPredictor(self.cfg)

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