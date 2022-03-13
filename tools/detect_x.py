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
# load config
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32*0.75, 64*0.75, 128*0.75, 256*0.75, 512*0.75]]
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Set threshold for this model
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
# cfg.MODEL.DEVICE='cpu'
#load trained weights
model = build_model(cfg) # returns a torch.nn.Module
cfg.MODEL.WEIGHTS = "./out_dir/parameters/object_segmentation/model_final.pth"    
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
model.train(False) # inference mode
# create predictor
predictor = DefaultPredictor(cfg)
model = build_model(cfg) # returns a torch.nn.Module
DetectionCheckpointer(model).load('./out_dir/parameters/object_segmentation/model_final.pth') # must load weights this way, can't use cfg.MODEL.WEIGHTS = "..."
model.train(False) # inference mode
img = cv2.imread("./datasets/object_images/dataset_05032022_test/12.jpg")
img = np.transpose(img,(2,0,1))
img_tensor = torch.from_numpy(img)
inputs = [{"image":img_tensor}, {"image":img_tensor}] # inputs is ready
start = time.time()
a = model(inputs)
end = time.time()
print("inference batch:", end-start)
print(np.shape(a))
# start = time.time()
# x1 = predictor(cv2.imread("./datasets/object_images/dataset_05032022_test/12.jpg"))
# x2 = predictor(cv2.imread("./datasets/object_images/dataset_05032022_test/12.jpg"))
# x3 = predictor(cv2.imread("./datasets/object_images/dataset_05032022_test/12.jpg"))

# end = time.time()
# print("inference:", end-start)



# return model,predictor