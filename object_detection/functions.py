#support functions for object detection
import torch 
from detectron2 import model_zoo
from detectron2.config import get_cfg

def yolo(weight):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight, device='cpu')
    model.conf = 0.4
    model.iou = 0.4

    
    return model
def cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("category_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    cfg.MODEL.WEIGHTS = "./out_dir/parameters/object_segmentation/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    cfg.DATASETS.TEST = ("skin_test", )
    cfg.MODEL.DEVICE='cpu'
    cfg.INPUT.MASK_FORMAT = 'bitmask'

    return cfg

