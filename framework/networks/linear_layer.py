import torch
from torch import nn

class BaseNetwork(nn.Module):
    def __init__(self, input_size: tuple, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        # TODO: Choice of activation function
        # TODO: choice of loss function
        # TODO: Choice of optimiser
        # TODO: batch norm, dropout, use biases?
        self.build_network()
    
    def _device(self):
        return next(self.parameters()).device
    
    def _is_on_cuda(self):
        next_ = next(self.parameters())
        return next_.is_cuda

    def loss(self):
        raise NotImplementedError

    def build_network(self):
        raise NotImplementedError
        
    def forward(self):
        raise NotImplementedError