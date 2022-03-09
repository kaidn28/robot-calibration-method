#support functions for object detection
import torch 

def yolo(weight):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight)
    model.conf = 0.4
    model.iou = 0.4
    
    return model