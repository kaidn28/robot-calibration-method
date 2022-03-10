# from detectron2 import model_zoo
# from detectron2.engine import DefaultTrainer, DefaultPredictor
# from detectron2.config import get_cfg
# import os 
# import cv2
# import numpy as np 
# from imantics import Polygons, Mask


# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("category_train",)
# cfg.DATASETS.TEST = ()
# cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.00025
# cfg.SOLVER.MAX_ITER = 1000
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

# cfg.MODEL.WEIGHTS = "./out_dir/parameters/object_segmentation/model_final.pth"
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
# cfg.DATASETS.TEST = ("skin_test", )
# cfg.MODEL.DEVICE='cpu'
# cfg.INPUT.MASK_FORMAT = 'bitmask'

# def maskRCNN(img, cfg):
#     predictor = DefaultPredictor(cfg)

#     outputs = predictor(img)
#     # print(outputs['instances'].pred_masks[0])
#     masks = outputs['instances'].pred_masks
#     masks = masks.int()
#     o = []
#     for i, mask in enumerate(masks):
#         polygons = Mask(np.array(masks[0])).polygons()
#         print(outputs['instances'].pred_classes.tolist()[i] == 0)
#         if outputs['instances'].pred_classes.tolist()[i] == 0:
#             o.append({"class_name": "black", "polygon": polygons.points})
#         elif outputs['instances'].pred_classes.tolist()[i] == 1:
#             o.append({"class_name": "white", "polygon": polygons.points})
#         elif outputs['instances'].pred_classes.tolist()[i] == 2:
#             o.append({"class_name": "silver", "polygon": polygons.points})

#     return o 


# img = cv2.imread("./datasets/object_images/dataset_03092022/12.jpg")
# print(maskRCNN(img, cfg))

# # print(np.shape(polygons.points))
# # print(masks[0].shape)


# # cv2.imshow("a", img)
# # cv2.waitKey()
import sys
import os
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from object_detection import segment
from object_detection.functions import cfg

cfg = cfg()
a = segment.ObjectSegment(cfg)
img = cv2.imread("./datasets/object_images/dataset_03092022/12.jpg")

res = a.predict(img)
print(res)