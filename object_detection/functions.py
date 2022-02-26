#support functions for object detection
import torch 

def yolo(image, weight):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight)
    model.conf = 0.4
    model.iou = 0.4
    img = image[..., ::-1]
    results = model(img)
    results.print()
    a = results.pandas().xywh[0]
    return a 