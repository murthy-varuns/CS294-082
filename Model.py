from torchvision import models
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from torch import optim

class model():
    
    def __init__(self, version, pretrained):
        """
        Initializes a model to a VGG instance with parameters for the following tasks:
        1. Training
        2. Classification
        3. Optimization
        4. Dataset Loading
        5. Capacity Estimation
        """
        assert(type(version) == str and type(pretrained) == bool)
        
        # This project was initially developed for VGG16Net.
        #from torchvision import models
        if version=='vgg16':
            self.model = models.vgg16(pretrained=pretrained)
        else:
            # Initialize other VGGs below.
            pass
        
        # Set training parameters.
        self.num_epochs = 3
        self.max_epochs_stop = 3
        self.num_classes = 10
        self.batch_size = 100
        self.learning_rate = 0.001
        self.print_every = 2
        
        # Freeze model weights.
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Add on classifier.
        n_inputs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Sequential(
                              nn.Linear(n_inputs, 256), 
                              nn.ReLU(), 
                              nn.Dropout(0.4),
                              nn.Linear(256, 10),                   
                              nn.LogSoftmax(dim=1))
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        # Add an optimizer.
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        
        # Set dataset loaders and capacity estimator to None for now.
        self.train_loader = None
        self.test_loader = None
        self.valid_loader = None
        self.capacity_estimator = None
        
    def set_capacity_estimator(self, capacity_estimator):
        self.capacity_estimator = capacity_estimator
    
    def show_trainable_params(self):
        """
        Shows the number of total parameters and trainable parameters as a complexity estimator.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Total Parameters: {total_params:,}')
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Training Parameters: {total_trainable_params:,}')
        
    
    def parallelize(self, cuda=False):
        """
        Moves and distributes the model across available GPUs.
        """
        if cuda:
            self.model = self.model.to('cuda')
        self.model = nn.DataParallel(self.model)
        
    def prepare_dataset(self, dataset='MNIST'):
        """
        Very self-explanatory, yeah?
        """
        # Define transformations.
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
        _normalize = transforms.Normalize((0.5,), (0.5,))

        # Load datasets.
        if dataset=='MNIST':
            train_dataset = torchvision.datasets.MNIST(root='/data/', train=True, transform=transforms.ToTensor(), download=True)
            valid_dataset = torchvision.datasets.MNIST(root='/data/', train=True, transform=transforms.Compose([transforms.ToTensor(), _normalize]), download=True)
            test_dataset = torchvision.datasets.MNIST(root='/data/', train=False, transform=transforms.ToTensor())
        else:
            pass
        
        # Set loaders.
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)