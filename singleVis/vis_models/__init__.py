'''https://github.com/AntixK/PyTorch-VAE'''
from .base import BaseVisModel
from .norm_ae import BN_AE, LN_AE, IN_AE
from .cn import CN_AE

vis_models = {
    'AE': BaseVisModel,
    'AEBatchNorm': BN_AE,
    'AELayerNorm': LN_AE,
    'AEInstanceNorm': IN_AE,
    "AEContinualNorm": CN_AE
}
