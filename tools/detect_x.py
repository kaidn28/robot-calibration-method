import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
import cv2
from object_detection import ObjectDetector

a = ObjectDetector("./out_dir/train/detection/center.pt")
# print(a.predict(cv2.imread('./datasets/object_images/dataset_29072021/img_4.jpg')))
img = cv2.imread('./datasets/object_images/dataset_29072021/img_4.jpg')
res = a.predict(img)
print(res[res["name"]=="x"])