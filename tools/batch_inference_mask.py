import torch
import os 
import sys 
from detectron2.engine import DefaultTrainer, DefaultPredictor
import cv2 

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from object_detection.functions import cfg
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo
import numpy as np 
import time

cfg = cfg()
model = build_model(cfg)
cfg.MODEL.WEIGHTS = "./out_dir/parameters/object_segmentation/model_final.pth"    
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
model.train(False) # inference mode
DetectionCheckpointer(model).load('./out_dir/parameters/object_segmentation/model_final.pth')
def inf_batch(model, folder, nums_batch):
    # folder = "./datasets/object_images/dataset_05032022_test"
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)

    res = []
    count = 0
    print(len(images))
    for i in range(0,len(images), nums_batch):
        print(i)
        list_tensor = []
        for j in range(nums_batch):
            images[i+j] = np.transpose(images[i+j], (2,0,1))
            images[i+j]= torch.from_numpy(images[i+j])
            list_tensor.append({"image" : images[i+j]})
        # print(list_tensor)
        a = model(list_tensor)
        res.append(a)
        print("done batch ", count + 1)
        count += 1
            
    return res 

# img = cv2.imread("./datasets/object_images/dataset_05032022_test/12.jpg")
# img = np.transpose(img,(2,0,1))
# img_tensor = torch.from_numpy(img)
# inputs = [{"image":img_tensor}, {"image":img_tensor},{"image":img_tensor}]
start = time.time()
res = inf_batch(model, "./datasets/object_images/dataset_05032022_test", 2)
end = time.time()
print(end-start)