########################################################################################################################
#                                                          IMPORT                                                      #
########################################################################################################################
import torch
import sys
import os
import json
import time
import numpy as np
import argparse

from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from umap.umap_ import find_ab_params

from singleVis.custom_weighted_random_sampler import CustomWeightedRandomSampler
from singleVis.SingleVisualizationModel import VisModel
from singleVis.losses import UmapLoss, ReconstructionLoss, TemporalLoss, DVILoss, SingleVisLoss, DummyTemporalLoss
from singleVis.edge_dataset import DVIDataHandler
from singleVis.trainer import DVITrainer
from singleVis.data import NormalDataProvider
from singleVis.subsampling import IdentitySampling
from singleVis.spatial_edge_constructor import SingleEpochSpatialEdgeConstructor
from singleVis.projector import DVIProjector
from singleVis.eval.evaluator import Evaluator
from singleVis.visualizer import visualizer
from singleVis.utils import find_neighbor_preserving_rate

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
data_provider = NormalDataProvider(CONTENT_PATH, net, EPOCH_START, EPOCH_END, EPOCH_PERIOD, device=DEVICE, classes=CLASSES, epoch_name=EPOCH_NAME, verbose=1)
if PREPROCESS:
    data_provider._meta_data(batch_size=SAVE_BATCH_SIZE)
    if B_N_EPOCHS >0:
        data_provider._estimate_boundary(LEN//10, l_bound=L_BOUND, batch_size=SAVE_BATCH_SIZE)

# Define visualization models
model = VisModel(ENCODER_DIMS, DECODER_DIMS)

# Define Losses
negative_sample_rate = 5
min_dist = .1
_a, _b = find_ab_params(1.0, min_dist)
umap_loss_fn = UmapLoss(negative_sample_rate, _a, _b, repulsion_strength=1.0)
recon_loss_fn = ReconstructionLoss(beta=1.0)
single_loss_fn = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA1)
# Define Projector
projector = DVIProjector(vis_model=model, content_path=CONTENT_PATH, vis_model_name=VIS_MODEL_NAME, epoch_name=EPOCH_NAME, device=DEVICE)

vis = visualizer(data_provider, projector, 200, "tab10")
save_dir = os.path.join(data_provider.content_path, "img")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
evaluator = Evaluator(data_provider, projector, metric=METRIC)

start_flag = 1
prev_model = VisModel(ENCODER_DIMS, DECODER_DIMS)

for iteration in range(EPOCH_START, EPOCH_END+EPOCH_PERIOD, EPOCH_PERIOD):
    # Define DVI Loss
    if start_flag:
        temporal_loss_fn = DummyTemporalLoss(DEVICE)
        criterion = DVILoss(umap_loss_fn, recon_loss_fn, temporal_loss_fn, lambd1=LAMBDA1, lambd2=0.0)
        start_flag = 0
    else:
        # TODO AL mode, redefine train_representation
        prev_data = data_provider.train_representation(iteration-EPOCH_PERIOD)
        curr_data = data_provider.train_representation(iteration)
        npr = find_neighbor_preserving_rate(prev_data, curr_data, N_NEIGHBORS, METRIC)
        temporal_loss_fn = TemporalLoss(w_prev, DEVICE)
        criterion = DVILoss(umap_loss_fn, recon_loss_fn, temporal_loss_fn, lambd1=LAMBDA1, lambd2=LAMBDA2*npr.mean())
    # Define training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)
    # Define Edge dataset
    t0 = time.time()
    sampler = IdentitySampling()
    spatial_cons = SingleEpochSpatialEdgeConstructor(data_provider, iteration, S_N_EPOCHS, B_N_EPOCHS, N_NEIGHBORS, metric=METRIC, sampler=sampler)
    edge_to, edge_from, probs, feature_vectors, attention = spatial_cons.construct()
    t1 = time.time()

    probs = probs / (probs.max()+1e-3)
    eliminate_zeros = probs>5e-2    #1e-3
    edge_to = edge_to[eliminate_zeros]
    edge_from = edge_from[eliminate_zeros]
    probs = probs[eliminate_zeros]
    
    dataset = DVIDataHandler(edge_to, edge_from, feature_vectors, attention)

    n_samples = int(np.sum(S_N_EPOCHS * probs) // 1)
    # chose sampler based on the number of dataset
    if len(edge_to) > pow(2,24):
        sampler = CustomWeightedRandomSampler(probs, n_samples, replacement=True)
    else:
        sampler = WeightedRandomSampler(probs, n_samples, replacement=True)
    edge_loader = DataLoader(dataset, batch_size=2000, sampler=sampler, num_workers=8, prefetch_factor=10)

    ########################################################################################################################
    #                                                       TRAIN                                                          #
    ########################################################################################################################

    trainer = DVITrainer(model, criterion, optimizer, lr_scheduler, edge_loader=edge_loader, DEVICE=DEVICE)

    t2=time.time()
    trainer.train(PATIENT, MAX_EPOCH)
    t3 = time.time()

    # save result
    save_dir = data_provider.model_path
    trainer.record_time(save_dir, "time_{}".format(VIS_MODEL_NAME), "complex_construction", str(iteration), t1-t0)
    trainer.record_time(save_dir, "time_{}".format(VIS_MODEL_NAME), "training", str(iteration), t3-t2)
    save_dir = os.path.join(data_provider.model_path, "Epoch_{}".format(iteration))
    trainer.save(save_dir=save_dir, file_name="{}".format(VIS_MODEL_NAME))

    print("Finish epoch {}...".format(iteration))

    vis.savefig(iteration, "{}_{}.png".format(VIS_METHOD, iteration))
    evaluator.save_epoch_eval(iteration, 15, temporal_k=5, file_name="{}".format(EVALUATION_NAME))

    prev_model.load_state_dict(model.state_dict())
    for param in prev_model.parameters():
        param.requires_grad = False
    w_prev = dict(prev_model.named_parameters())
