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
# from singleVis.SingleVisualizationModel import VisModel
from singleVis.vis_models import BN_AE
from singleVis.losses import UmapLoss, ReconstructionLoss, SingleVisLoss, LocalTemporalLoss, SmoothnessLoss
from singleVis.edge_dataset import DVIDataHandler, LocalTemporalDataHandler
from singleVis.trainer import DVITrainer, SingleVisTrainer, LocalTemporalTrainer
from singleVis.data import NormalDataProvider
from singleVis.spatial_edge_constructor import LocalSpatialTemporalEdgeConstructor, SingleEpochSpatialEdgeConstructor
from singleVis.projector import DVIProjector
from singleVis.eval.evaluator import Evaluator
from singleVis.visualizer import visualizer
########################################################################################################################
#                                                     DVI PARAMETERS                                                   #
########################################################################################################################
"""DVI with semantic temporal edges"""
VIS_METHOD = "tDVI" 

########################################################################################################################
#                                                     LOAD PARAMETERS                                                  #
########################################################################################################################
parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', type=str)
args = parser.parse_args()

CONTENT_PATH = args.content_path
sys.path.append(CONTENT_PATH)
with open(os.path.join(CONTENT_PATH, "config.json"), "r") as f:
    config = json.load(f)
config = config[VIS_METHOD]

SETTING = config["SETTING"]
CLASSES = config["CLASSES"]
DATASET = config["DATASET"]
PREPROCESS = config["VISUALIZATION"]["PREPROCESS"]
GPU_ID = config["GPU"]
EPOCH_START = config["EPOCH_START"]
EPOCH_END = config["EPOCH_END"]
EPOCH_PERIOD = config["EPOCH_PERIOD"]
EPOCH_NAME = config["EPOCH_NAME"]

# Training parameter (subject model)
TRAINING_PARAMETER = config["TRAINING"]
NET = TRAINING_PARAMETER["NET"]
LEN = TRAINING_PARAMETER["train_num"]

# Training parameter (visualization model)
VISUALIZATION_PARAMETER = config["VISUALIZATION"]
LAMBDA1 = VISUALIZATION_PARAMETER["LAMBDA1"]
B_N_EPOCHS = VISUALIZATION_PARAMETER["BOUNDARY"]["B_N_EPOCHS"]
L_BOUND = VISUALIZATION_PARAMETER["BOUNDARY"]["L_BOUND"]
ENCODER_DIMS = VISUALIZATION_PARAMETER["ENCODER_DIMS"]
DECODER_DIMS = VISUALIZATION_PARAMETER["DECODER_DIMS"]
S_N_EPOCHS = VISUALIZATION_PARAMETER["S_N_EPOCHS"]
T_N_EPOCHS = VISUALIZATION_PARAMETER["T_N_EPOCHS"]
N_NEIGHBORS = VISUALIZATION_PARAMETER["N_NEIGHBORS"]
PATIENT = VISUALIZATION_PARAMETER["PATIENT"]
MAX_EPOCH = VISUALIZATION_PARAMETER["MAX_EPOCH"]

VIS_MODEL_NAME = VISUALIZATION_PARAMETER["VIS_MODEL_NAME"]
EVALUATION_NAME = VISUALIZATION_PARAMETER["EVALUATION_NAME"]

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
    data_provider._meta_data()
    if B_N_EPOCHS >0:
        data_provider._estimate_boundary(LEN//10, l_bound=L_BOUND)

# Define visualization models
# model = VisModel(ENCODER_DIMS, DECODER_DIMS)
model = BN_AE(ENCODER_DIMS, DECODER_DIMS)

# Define Losses/home/xianglin/projects/DVI_data/resnet18_mnist
negative_sample_rate = 5
min_dist = .1
_a, _b = find_ab_params(1.0, min_dist)
umap_loss_fn = UmapLoss(negative_sample_rate, DEVICE, _a, _b, repulsion_strength=1.0)
recon_loss_fn = ReconstructionLoss(beta=1.0)
smooth_loss_fn = SmoothnessLoss(margin=0.0)
single_loss_fn = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA1)
# Define Projector
projector = DVIProjector(vis_model=model, content_path=CONTENT_PATH, vis_model_name=VIS_MODEL_NAME, epoch_name=EPOCH_NAME, device=DEVICE)

vis = visualizer(data_provider, projector, 200, "tab10")
save_dir = os.path.join(data_provider.content_path, "img")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
evaluator = Evaluator(data_provider, projector, metric="euclidean")

# start epoch
# Define Criterion
criterion = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA1)
# Define training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)
# Define Edge dataset
t0 = time.time()

spatial_cons = SingleEpochSpatialEdgeConstructor(data_provider, EPOCH_START, S_N_EPOCHS, B_N_EPOCHS, N_NEIGHBORS, metric="euclidean")
edge_to, edge_from, probs, feature_vectors, attention = spatial_cons.construct()

dataset = DVIDataHandler(edge_to, edge_from, feature_vectors, attention)

n_samples = int(np.sum(S_N_EPOCHS * probs) // 1)
# chose sampler based on the number of dataset
if len(edge_to) > pow(2,24):
    sampler = CustomWeightedRandomSampler(probs, n_samples, replacement=True)
else:
    sampler = WeightedRandomSampler(probs, n_samples, replacement=True)
edge_loader = DataLoader(dataset, batch_size=1000, sampler=sampler, num_workers=4, prefetch_factor=10)

# train
trainer = SingleVisTrainer(model, criterion, optimizer, lr_scheduler,edge_loader=edge_loader, DEVICE=DEVICE)
t2=time.time()
trainer.train(PATIENT, MAX_EPOCH)
t3 = time.time()

save_dir = os.path.join(data_provider.model_path, "Epoch_{}".format(EPOCH_START))
trainer.save(save_dir=save_dir, file_name="{}".format(VIS_MODEL_NAME))
save_dir = os.path.join(data_provider.content_path, "img")
vis.savefig(EPOCH_START, path=os.path.join(save_dir, "{}_{}_{}.png".format(DATASET, EPOCH_START, VIS_METHOD)))
evaluator.save_epoch_eval(EPOCH_START, 15, temporal_k=5, file_name="{}".format(EVALUATION_NAME))
# get prev_embedding
prev_embedding = projector.batch_project(EPOCH_START, data_provider.train_representation(EPOCH_START))

for iteration in range(EPOCH_START+EPOCH_PERIOD, EPOCH_END+EPOCH_PERIOD, EPOCH_PERIOD):
    # Define Criterion
    criterion = LocalTemporalLoss(umap_loss_fn, recon_loss_fn, smooth_loss_fn, lambd=LAMBDA1)
    # Define training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)
    # Define Edge dataset
    t0 = time.time()
    spatial_cons = LocalSpatialTemporalEdgeConstructor(data_provider, S_N_EPOCHS, B_N_EPOCHS, T_N_EPOCHS, N_NEIGHBORS, metric="euclidean")
    edge_to, edge_from, probs, feature_vectors, attention, coefficient, embedded = spatial_cons.construct(iteration-EPOCH_PERIOD, iteration, prev_embedding)
    t1 = time.time()
    
    dataset = LocalTemporalDataHandler(edge_to, edge_from, feature_vectors, attention, coefficient, embedded)

    n_samples = int(np.sum(S_N_EPOCHS * probs) // 1)
    # chose sampler based on the number of dataset
    if len(edge_to) > pow(2,24):
        sampler = CustomWeightedRandomSampler(probs, n_samples, replacement=True)
    else:
        sampler = WeightedRandomSampler(probs, n_samples, replacement=True)
    edge_loader = DataLoader(dataset, batch_size=1000, sampler=sampler, num_workers=4, prefetch_factor=10, pin_memory=True)

    ########################################################################################################################
    #                                                       TRAIN                                                          #
    ########################################################################################################################

    trainer = LocalTemporalTrainer(model, criterion, optimizer, lr_scheduler,edge_loader=edge_loader, DEVICE=DEVICE)

    t2=time.time()
    trainer.train(PATIENT, MAX_EPOCH)
    t3 = time.time()

    save_dir = os.path.join(data_provider.model_path, "Epoch_{}".format(iteration))
    trainer.save(save_dir=save_dir, file_name="{}".format(VIS_MODEL_NAME))

    save_dir = os.path.join(data_provider.content_path, "img")
    vis.savefig(iteration, path=os.path.join(save_dir, "{}_{}_{}.png".format(DATASET, iteration, VIS_METHOD)))
    evaluator.save_epoch_eval(iteration, 15, temporal_k=5, file_name="{}".format(EVALUATION_NAME))

    # get prev_embedding
    prev_embedding = projector.batch_project(iteration, data_provider.train_representation(iteration))

