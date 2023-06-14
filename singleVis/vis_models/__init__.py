'''https://github.com/AntixK/PyTorch-VAE'''
from .base import BaseVisModel
from .bn_ae import BN_AE

vis_models = {
    'AE': BaseVisModel,
    'AEBatchNorm': BN_AE 
}
