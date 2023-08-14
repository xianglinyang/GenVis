from yacs.config import CfgNode as CN
import argparse

config_file = "/home/xianglin/projects/DVI_data/resnet18_mnist/config/tdvi.yaml"


cfg = CN()
cfg.ENCODER_DIMS = []
cfg.VIS_MODEL = None
cfg.VIS_MODEL_NAME = "None"
cfg.merge_from_file(config_file)

print(1)
