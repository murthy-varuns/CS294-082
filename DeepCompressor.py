import torch
import numpy as np
import pandas as pd
from Model import model
from Trainer import trainer
from CapacityEstimator import capacity_estimator

class deep_compressor():
    """
    A DeepCompressor is used to stochastically reduce parameters of a neural network. Here, we use
    VGG16 to illustrate the Generalization-Capacity ratio while training VGG16 on MNIST.     
    """
    
    def __init__(self, model, trainer, capacity_estimator=None):
        """
        Initializes a DeepCompressor with a Model, Trainer & CapacityEstimator.
        """
        # Add the Model and its Trainer to the DeepCompressor.
        self.model = model
        self.trainer = trainer
        # Add a new CapacityEstimator or use the Model's one.
        
        assert(self.model.capacity_estimator or capacity_estimator, 'We need atleast 1 CapacityEstimator!')
        if capacity_estimator:
            self.capacity_estimator = capacity_estimator
        else:
            self.capacity_estimator = model.capacity_estimator

        
    def train(self, dropout=0):
        """
        Trains Model with Trainer's train function with an optional 
        dropout by deactivating random neurons of the Model's linear layers.
        """
        assert(dropout >= 0)
    
        
        if dropout > 0:
            copy = torch.nn.parameter.Parameter(self.model.model.module.classifier[0].weight.clone(), requires_grad=True)
            rows, _ = copy.shape
            for i in np.random.choice(rows, dropout):
                self.model.model.module.classifier[0].weight[i] = torch.zeros(25088)
                    
            self.trainer.train()
            self.model.model.module.classifier[0].weight = copy
        else:
            self.trainer.train()
        
    def test(self):
        """
        Tests Model with Trainer's test function.
        """
        self.trainer.test()
        
    def squeeze(self, n=10, norm=None):
        """
        Prunes parameters from linear layers by L1 norm, L2 norm, reservoir 
        sampling (stochastic), or simply choosing the smallest one(s).
        
        VGG16 has the following fully-connected architecture (scaled down for visual purposes):
        
        Layer #:   1       2      3     4     5 
                 (25088) (4096) (4096) (256) (10)
                    x
                    x
                    x      x       x     
                    x      x       x     
                    .      .       .     x
                    .      .       .     .     x
                    .      .       .     .     x
                    .      .       .     x     
                    x      x       x     
                    x      x       x
                    x
                    x
            
        We can only drop neurons on the layers 2, 3 & 4 (the layers with 4096, 4096 and 256 neurons). 
        """
        if norm:
            assert(norm=='mean-l1' or norm=='mean-l2' or norm=='reservoir')
        
        # Get the weights of the first hidden layer.
        weights = self.model.model.module.classifier[0].weight
        weights.requires_grad = False
        shape = weights.shape
        assert(shape[0] == 4096 and shape[1] == 25088)
        
        # Calculate the sum of the weights of the connections for each of the 4096 neurons in the first layer.
        weights_sum = torch.sum(weights, dim=1)
        assert(weights_sum.shape[0] == 4096)
        
        if norm=='mean-l1':
            # L1 norm
            to_remove = torch.mul((weights_sum - weights_sum.mean()).abs_(), -1).topk(n)[1]
        elif norm=='mean-l2':
            # Not implemented yet.
            pass
        elif norm=='reservoir':
            # Not implemented yet.
            pass
        else:
            # Simple removal of smallest n neurons.
            to_remove = torch.mul(weights_sum, -1).topk(n)[1]
        self.model.model.module.classifier[0].weight[to_remove] = torch.zeros(25088).cuda()
    