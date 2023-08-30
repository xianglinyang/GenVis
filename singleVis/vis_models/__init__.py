'''https://github.com/AntixK/PyTorch-VAE'''
from .base import AE
from .batch_norm import BatchNormAE, BatchNormBaseAE
from .layer_norm import LayerNormAE
from .instance_norm import InstanceNormAE
from .group_norm import GroupNormAE
from .continual_norm import ContinualNormAE

vis_models = {
    'AE': AE,
    'bnAE': BatchNormAE,
    'baseAE': BatchNormBaseAE,
    'lnAE': LayerNormAE,
    'inAE': InstanceNormAE,
    'gnAE': GroupNormAE,
    'cnAE': ContinualNormAE
}
