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
from singleVis.vis_models import vis_models as vmodels
from singleVis.losses import UmapLoss, ReconstructionLoss, SingleVisLoss
from singleVis.edge_dataset import DataHandler
from singleVis.trainer import SingleVisTrainer
from singleVis.data import NormalDataProvider
from singleVis.spatial_edge_constructor import SingleEpochSpatialEdgeConstructor
from singleVis.projector import DVIProjector
from singleVis.eval.evaluator import Evaluator
from singleVis.subsampling import DensityAwareSampling
###################################################################################
# from pynndescent import NNDescent
# def density_estimation(train_data, k=10, metric="euclidean"):
#     # number of trees in random projection forest
#     n_trees = min(64, 5 + int(round((train_data.shape[0]) ** 0.5 / 20.0)))
#     # max number of nearest neighbor iters to perform
#     n_iters = max(5, int(round(np.log2(train_data.shape[0]))))
#     # distance metric
#     # get nearest neighbors
#     nnd = NNDescent(
#         train_data,
#         n_neighbors=k,
#         metric=metric,
#         n_trees=n_trees,
#         n_iters=n_iters,
#         max_candidates=60,
#         verbose=False
#     )
#     _, knn_dists = nnd.neighbor_graph
#     avg_distance = knn_dists[:, -1]
#     density = k / avg_distance
#     return avg_distance.mean(), density

########################################################################################################################
#                                                     DVI PARAMETERS                                                   #
########################################################################################################################
"""This serve as an example of DeepVisualInsight implementation in pytorch."""
VIS_METHOD = "singleDVI" # DeepVisualInsight

########################################################################################################################
#                                                     LOAD PARAMETERS                                                  #
########################################################################################################################
parser = argparse.ArgumentParser(description='Process hyperparameters...')
parser.add_argument('--content_path', '-c', type=str)
parser.add_argument('--iteration','-i', type=int)
parser.add_argument('-g', type=int, default=0)
parser.add_argument('-r', help='ratio', type=float)
parser.add_argument('--method', "-m", type=str)
args = parser.parse_args()

CONTENT_PATH = args.content_path
I = args.iteration
GPU_ID = args.g
RATIO = args.r
M = args.method

sys.path.append(CONTENT_PATH)
with open(os.path.join(CONTENT_PATH, "config.json"), "r") as f:
    config = json.load(f)
config = config[VIS_METHOD]

# record output information
# now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time())) 
# sys.stdout = open(os.path.join(CONTENT_PATH, now+".txt"), "w")

SETTING = config["SETTING"]
CLASSES = config["CLASSES"]
DATASET = config["DATASET"]
PREPROCESS = config["VISUALIZATION"]["PREPROCESS"]

# Training parameter (subject model)
TRAINING_PARAMETER = config["TRAINING"]
NET = TRAINING_PARAMETER["NET"]
LEN = TRAINING_PARAMETER["train_num"]
EPOCH_START = config["EPOCH_START"]
EPOCH_END = config["EPOCH_END"]
EPOCH_PERIOD = config["EPOCH_PERIOD"]
EPOCH_NAME = config["EPOCH_NAME"]

# Training parameter (visualization model)
VISUALIZATION_PARAMETER = config["VISUALIZATION"]
VIS_MODEL = VISUALIZATION_PARAMETER["VIS_MODEL"]
LAMBDA = VISUALIZATION_PARAMETER["LAMBDA"]
B_N_EPOCHS = VISUALIZATION_PARAMETER["BOUNDARY"]["B_N_EPOCHS"]
L_BOUND = VISUALIZATION_PARAMETER["BOUNDARY"]["L_BOUND"]
ENCODER_DIMS = VISUALIZATION_PARAMETER["ENCODER_DIMS"]
DECODER_DIMS = VISUALIZATION_PARAMETER["DECODER_DIMS"]
S_N_EPOCHS = VISUALIZATION_PARAMETER["S_N_EPOCHS"]
N_NEIGHBORS = VISUALIZATION_PARAMETER["N_NEIGHBORS"]
PATIENT = VISUALIZATION_PARAMETER["PATIENT"]
MAX_EPOCH = VISUALIZATION_PARAMETER["MAX_EPOCH"]

VIS_MODEL_NAME = VISUALIZATION_PARAMETER["VIS_MODEL_NAME"]
EVALUATION_NAME = VISUALIZATION_PARAMETER["EVALUATION_NAME"]
VIS_MODEL_NAME = f"{VIS_MODEL_NAME}_{RATIO}_{M}"
EVALUATION_NAME = f"{EVALUATION_NAME}_{RATIO}_{M}"

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
model = vmodels[VIS_MODEL](ENCODER_DIMS, DECODER_DIMS)

# Define Losses
negative_sample_rate = 5
min_dist = .1
_a, _b = find_ab_params(1.0, min_dist)
umap_loss_fn = UmapLoss(negative_sample_rate, _a, _b, repulsion_strength=1.0)
recon_loss_fn = ReconstructionLoss(beta=1.0)
# Define Projector
projector = DVIProjector(vis_model=model, content_path=CONTENT_PATH, vis_model_name=VIS_MODEL_NAME, epoch_name=EPOCH_NAME, device=DEVICE)

# Define DVI Loss
criterion = SingleVisLoss(umap_loss_fn, recon_loss_fn, lambd=LAMBDA)

# Define training parameters
optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)
# Define Edge dataset
t0 = time.time()
spatial_cons = SingleEpochSpatialEdgeConstructor(data_provider, I, S_N_EPOCHS, B_N_EPOCHS, N_NEIGHBORS, metric="euclidean")

# sampling with different strategies
train_data = data_provider.train_representation(I)

# das = DensityAwareSampling(train_data, n_classes=15, k=10, metric="euclidean")
# selected = das.sampling(ratio=RATIO, temperature=.5)
selected = np.random.choice(len(train_data), 5993, replace=False)
train_data = train_data[selected]

edge_to, edge_from, probs, feature_vectors, attention = spatial_cons.construct(train_data)
t1 = time.time()

dataset = DataHandler(edge_to, edge_from, feature_vectors, attention)

n_samples = int(np.sum(S_N_EPOCHS * probs) // 1)
# chose sampler based on the number of dataset
if len(edge_to) > pow(2,24):
    sampler = CustomWeightedRandomSampler(probs, n_samples, replacement=True)
else:
    sampler = WeightedRandomSampler(probs, n_samples, replacement=True)
edge_loader = DataLoader(dataset, batch_size=1000, sampler=sampler, num_workers=4, prefetch_factor=10, pin_memory=True)

#######################################################################################################################
#                                                       TRAIN                                                         #
#######################################################################################################################

trainer = SingleVisTrainer(model, criterion, optimizer, lr_scheduler,edge_loader=edge_loader, DEVICE=DEVICE)

t2=time.time()
trainer.train(PATIENT, MAX_EPOCH)
t3 = time.time()

# save time result
save_dir = data_provider.model_path
file_name = "time_{}".format(VIS_MODEL_NAME)
save_file = os.path.join(save_dir, file_name+".json")
if not os.path.exists(save_file):
    evaluation = dict()
else:
    f = open(save_file, "r")
    evaluation = json.load(f)
    f.close()
if "complex_construction" not in evaluation.keys():
    evaluation["complex_construction"] = dict()
evaluation["complex_construction"][str(I)] = round(t1-t0, 3)
if "training" not in evaluation.keys():
    evaluation["training"] = dict()
evaluation["training"][str(I)] = round(t3-t2, 3)
with open(save_file, 'w') as f:
    json.dump(evaluation, f)

save_dir = os.path.join(data_provider.model_path, "Epoch_{}".format(I))
trainer.save(save_dir=save_dir, file_name="{}".format(VIS_MODEL_NAME))

########################################################################################################################
#                                                      VISUALIZATION                                                   #
########################################################################################################################

from singleVis.visualizer import visualizer

vis = visualizer(data_provider, projector, 200, "tab10")
save_dir = os.path.join(data_provider.content_path, "img")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
# vis.savefig(I, path=os.path.join(save_dir, "{}_{}_{}_{}.png".format(VIS_METHOD, VIS_MODEL, I, ratio)))
pred = data_provider.get_pred(I, train_data).argmax(axis=1)
labels = data_provider.train_labels(I)[selected]
vis.savefig_cus(I, train_data, pred, labels, path=os.path.join(save_dir, "{}_{}_{}_{}.png".format(VIS_METHOD, VIS_MODEL, I, RATIO)))


########################################################################################################################
#                                                       EVALUATION                                                     #
########################################################################################################################
evaluator = Evaluator(data_provider, projector, metric="euclidean")
evaluator.save_epoch_eval(I, 15, temporal_k=5, file_name="{}".format(EVALUATION_NAME))
