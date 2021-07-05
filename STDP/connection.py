import torch
import torch.nn as nn


class BaseConnection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    def init_param(self, weight_initialization):
        # Initialize the learnable parameters in a network

        if weight_initialization == 'gaussian':
            self.w[:] = torch.normal(mean=0, std=1, size= self.w.size())
        elif weight_initialization == 'uniform':
            self.w[:] = torch.rand(self.w.size())
        elif isinstance(weight_initialization, float):
            self.w[:] = torch.full(self.w.size(), weight_initialization)
        elif torch.is_tensor(weight_initialization):
            self.w[:] = weight_initialization



class Connection(BaseConnection):
    # This Connection class is designed to connect between two 1D Population instances
    def __init__(self, source, target, learning=None, mode='m2m', weight_initialization='gaussian', synapse='cuba'):
        super().__init__()
        # Now, both source and target must be 1D Population instances
        self.source = source
        self.target = target
        self.synapse = synapse
        # learning must take a callable as an argument
        if learning:
            self.learning = learning(self)
        else:
            self.learning = learning
        # mode: m2m (many-to-many)
        #       o2o (one-to-one)
        #       m2o (many-to-one)
        #       o2m (one-to-many)
        self.mode = mode

        if mode == 'm2m':
            self.register_buffer('w', torch.zeros((target.neurons, source.neurons)))
        elif mode == 'o2o':
            if target.neurons == source.neurons:
                self.register_buffer('w', torch.zeros((target.neurons,)))
            else:
                raise Exception
            pass
        elif mode == 'm2o':
            pass
        elif mode == 'o2m':
            pass

        self.init_param(weight_initialization=weight_initialization)

    def forward(self, x):
        if self.synapse =='cuba':
            return torch.matmul(self.w, x)


    def update(self):
        if self.learning:
            self.learning()

    def reset_trace(self):
        if self.learning:
            self.learning.reset()
