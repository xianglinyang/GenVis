########################################################################################################################
#                                                          IMPORT                                                      #
########################################################################################################################
import torch
import sys
import os
import json
import argparse

from singleVis.SingleVisualizationModel import VisModel
from singleVis.data import NormalDataProvider
from singleVis.projector import DVIProjector
from singleVis.eval.evaluator import Evaluator

from config import load_cfg
########################################################################################################################
#                                                     DVI PARAMETERS                                                   #
########################################################################################################################
"""This serve as an example of DeepVisualInsight implementation in pytorch."""
VIS_METHOD = "dvi" # DeepVisualInsight

########################################################################################################################
#                                                     LOAD PARAMETERS                                                  #
########################################################################################################################
parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
args = parser.parse_args()

CONTENT_PATH = args.content_path
sys.path.append(CONTENT_PATH)

config = load_cfg(os.path.join(CONTENT_PATH, "config", f"{VIS_METHOD}.yaml"))

SETTING = config.SETTING
CLASSES = config.CLASSES
DATASET = config.DATASET
PREPROCESS = config.VISUALIZATION.PREPROCESS
GPU_ID = config.GPU
EPOCH_START = config.EPOCH_START
EPOCH_END = config.EPOCH_END
EPOCH_PERIOD = config.EPOCH_PERIOD
EPOCH_NAME = config.EPOCH_NAME

# Training parameter (subject model)
TRAINING_PARAMETER = config.TRAINING
NET = TRAINING_PARAMETER.NET
LEN = TRAINING_PARAMETER.train_num

# Training parameter (visualization model)
VISUALIZATION_PARAMETER = config.VISUALIZATION
SAVE_BATCH_SIZE = VISUALIZATION_PARAMETER.SAVE_BATCH_SIZE
LAMBDA1 = VISUALIZATION_PARAMETER.LAMBDA1
LAMBDA2 = VISUALIZATION_PARAMETER.LAMBDA2
B_N_EPOCHS = VISUALIZATION_PARAMETER.BOUNDARY.B_N_EPOCHS
L_BOUND = VISUALIZATION_PARAMETER.BOUNDARY.L_BOUND
ENCODER_DIMS = VISUALIZATION_PARAMETER.ENCODER_DIMS
DECODER_DIMS = VISUALIZATION_PARAMETER.DECODER_DIMS
S_N_EPOCHS = VISUALIZATION_PARAMETER.S_N_EPOCHS
N_NEIGHBORS = VISUALIZATION_PARAMETER.N_NEIGHBORS
PATIENT = VISUALIZATION_PARAMETER.PATIENT
MAX_EPOCH = VISUALIZATION_PARAMETER.MAX_EPOCH
METRIC = VISUALIZATION_PARAMETER.METRIC

VIS_MODEL_NAME = f"{VIS_METHOD}"
EVALUATION_NAME = f"evaluation_{VIS_MODEL_NAME}"


# Define hyperparameters
DEVICE = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu")

import Model.model as subject_model
net = eval("subject_model.{}()".format(NET))

########################################################################################################################
#                                                    TRAINING SETTING                                                  #
########################################################################################################################
# Define data_provider
data_provider = NormalDataProvider(CONTENT_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, epoch_name=EPOCH_NAME, device=DEVICE, classes=CLASSES,verbose=1)
# if PREPROCESS:
#     data_provider._meta_data(batch_size=SAVE_BATCH_SIZE)
#     if B_N_EPOCHS >0:
#         data_provider._estimate_boundary(LEN//10, l_bound=L_BOUND, batch_size=SAVE_BATCH_SIZE)

# Define visualization models
model = VisModel(ENCODER_DIMS, DECODER_DIMS)

# Define Projector
projector = DVIProjector(vis_model=model, content_path=CONTENT_PATH, epoch_name=EPOCH_NAME, vis_model_name=VIS_MODEL_NAME, device=DEVICE)
########################################################################################################################
#                                                      VISUALIZATION                                                   #
########################################################################################################################
from singleVis.visualizer import visualizer
# vis = visualizer(data_provider, projector, 200, "tab10")
# save_dir = os.path.join(data_provider.content_path, "img")
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
# for i in range(EPOCH_START, EPOCH_END+EPOCH_PERIOD, EPOCH_PERIOD):
#     vis.savefig(i, "{}_{}.png".format(VIS_METHOD, i))

    
########################################################################################################################
#                                                       EVALUATION                                                     #
########################################################################################################################
eval_epochs = range(EPOCH_START, EPOCH_END+EPOCH_PERIOD, EPOCH_PERIOD)
if EPOCH_END==200:
    eval_epochs = [1, 50, 100, 150, 200]
else:
    eval_epochs = [1, 5, 10, 15, 20]

evaluator = Evaluator(data_provider, projector, metric=METRIC)

for eval_epoch in eval_epochs:
    evaluator.save_epoch_eval(n_epoch=eval_epoch, n_neighbors=15, temporal_k=5, loss_corr=True, file_name="{}".format(EVALUATION_NAME))
