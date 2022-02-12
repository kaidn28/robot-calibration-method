#support functions for object detection
import torch 

def yolo(image):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./model/super-best.pt')
    model.conf = 0.6
    model.iou = 0.6
    img = image[..., ::-1]
    results = model(img)
    a = results.pandas().xywh[0]
    return a 