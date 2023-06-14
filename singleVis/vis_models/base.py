from torch import nn
from abc import abstractmethod

class BaseVisModel(nn.Module):

    def __init__(self) -> None:
        super(BaseVisModel, self).__init__()
    
    def encode(self, input):
        raise NotImplementedError

    def decode(self, input):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs):
        pass

