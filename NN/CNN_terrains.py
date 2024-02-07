from torch import nn
import torch.nn.functional as F
import torch
from numpy.linalg import eig
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

class CNN(nn.Module):
    def __init__(self,res=[640,480],latent_dim=8,hidden_layers=128,seed=0,numChannels=1,lr=2e-4):
        super(CNN, self).__init__()
        torch.manual_seed(seed)

        self.CNN_stack = nn.Sequential(
            Conv2d(in_channels=1, out_channels=1,kernel_size=(5,5),stride=2),
            nn.LeakyReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=1, out_channels=1,kernel_size=(5,5),stride=2),
            nn.LeakyReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
            
        )
        self.flatten_input = nn.Sequential(
            nn.Linear(29*39, hidden_layers),
            nn.LeakyReLU(),
        )

        self.mu = nn.Sequential(

            nn.Linear(hidden_layers,latent_dim)
        )
        self.sigma = nn.Sequential(
          
            nn.Linear(hidden_layers,latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,hidden_layers),
            ReLU(),
            nn.Linear(hidden_layers,hidden_layers),
            ReLU(),
            nn.Linear(hidden_layers,res[0]*res[1]),
            nn.Sigmoid()
        )

        self.optimizer=self.configure_optimizers(lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        
    def configure_optimizers(self,lr=1e-4):
        return torch.optim.AdamW(self.parameters(), lr=lr)
    
    def forward(self, x):
        cnn_out = self.CNN_stack(x)
        logits = self.flatten_input(cnn_out.flatten(1))
        mu = self.mu(logits)
        logstd = torch.exp(self.sigma(logits)/2)

        return logits, mu, logstd







# rough = np.load('terrain_npy/rough.npy')
# slope = np.load('terrain_npy/slope.npy')
# stairs = np.load('terrain_npy/stairs.npy')

# # rough_rec=np.zeros((np.shape(rough)[0],np.shape(rough)[1],1000))
# # slope_rec=np.zeros((np.shape(rough)[0],np.shape(rough)[1],1000))
# # stairs_rec=np.zeros((np.shape(rough)[0],np.shape(rough)[1],1000))
# out_tensor=[]
# for i in range(10):
#     out_tensor.append(torch.tensor(rough+np.random.normal(0,0.01,size=(np.shape(rough)[0],np.shape(rough)[1])),dtype=torch.float))
#     out_tensor.append(torch.tensor(slope+np.random.normal(0,0.01,size=(np.shape(rough)[0],np.shape(rough)[1])),dtype=torch.float))
#     out_tensor.append(torch.tensor(stairs+np.random.normal(0,0.01,size=(np.shape(rough)[0],np.shape(rough)[1])),dtype=torch.float))
#     # rough_rec[:,:,i]=rough+np.random.normal(0,0.01,size=(np.shape(rough)[0],np.shape(rough)[1]))
#     # slope_rec[:,:,i]=slope+np.random.normal(0,0.01,size=(np.shape(rough)[0],np.shape(rough)[1]))
#     # stairs_rec[:,:,i]=stairs+np.random.normal(0,0.01,size=(np.shape(rough)[0],np.shape(rough)[1]))

# torch.save(out_tensor,('terrain_npy/training_data')) 