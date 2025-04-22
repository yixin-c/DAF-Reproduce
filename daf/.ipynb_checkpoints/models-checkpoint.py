import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from tqdm import tqdm
from scipy.io import loadmat
from torch.autograd import Variable
import numpy as np
import torch.utils.data as utils
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import random
import itertools
from mpl_toolkits.axes_grid1 import ImageGrid

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
#         return max(1.0, int(epoch > self.decay_start_epoch)*10.0)
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

    
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, init_depth, s_depth, bc=True, n_residual_blocks=9, domains=2, log=True):
        super(Generator, self).__init__()

        self.z_size = 17 if bc else 7
        #### encode
        ec = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, init_depth, 7),
                    nn.InstanceNorm2d(init_depth),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        ec += [  nn.Conv2d(init_depth, init_depth*2, 3, stride=2, padding=1),
                 nn.InstanceNorm2d(init_depth*2),
                 nn.ReLU(inplace=True) ]
        
        ec += [  nn.Conv2d(init_depth*2, init_depth*4-s_depth, 3, stride=2, padding=1),
                 nn.InstanceNorm2d(init_depth*4-s_depth),
                 nn.ReLU(inplace=True) ]
        in_features = init_depth*4-s_depth
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            ec += [ResidualBlock(in_features)]
        
        #### encode
        self.ec = nn.Sequential(*ec)
        # Embed layer
        self.s_depth = s_depth
        self.embed = nn.Embedding(domains, self.z_size*self.z_size*s_depth)
        self.z1 = nn.Conv2d(in_features, int((in_features + s_depth)/2), 5, stride=4)
        self.tconv1 = nn.ConvTranspose2d(int((in_features + s_depth)/2), int((in_features + s_depth)/2), kernel_size=5, stride=4, padding=1, output_padding=2)
        self.tconv2 = nn.ConvTranspose2d(int((in_features + s_depth)/2), in_features, kernel_size=1, stride=1)
        
        #### decode
        in_features = init_depth*4
        dc = []            
        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            dc += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                     nn.InstanceNorm2d(out_features),
                     nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        if log:
            dc += [  nn.ReflectionPad2d(3),
                     nn.Conv2d(init_depth, output_nc, 7),
                     nn.ReLU()]
        else:
            dc += [  nn.ReflectionPad2d(3),
                     nn.Conv2d(init_depth, output_nc, 7),
                     nn.Tanh() ]
        #### decode
        self.dc = nn.Sequential(*dc)
        
    def forward(self, x, s):
        x = self.ec(x)
        x = self.z1(x)
        z = x.view(x.shape[0], -1)
        x = self.tconv2(self.tconv1(x))
        ##
        s_x = self.embed(s).view(x.shape[0], self.s_depth, self.z_size, self.z_size)
        xs = torch.cat((x,s_x),dim=1)
        xt = self.dc(xs)
        return xt, z
    
class Discriminator(nn.Module):
    def __init__(self, input_nc, init_depth):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, init_depth, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(init_depth, init_depth*2, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(init_depth*2), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(init_depth*2, init_depth*4, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(init_depth*4), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(init_depth*4, init_depth*8, 4, padding=1),
                    nn.InstanceNorm2d(init_depth*8), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(init_depth*8, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class MLP(nn.Module):
    def __init__(self, n_traits, input_dim):
        super(MLP, self).__init__()
        latent_dim = 256
        self.fc1 = nn.Linear(input_dim, 512*2)
        self.fc2 = nn.Linear(512*2,256*2)
        self.fc3 = nn.Linear(256*2,latent_dim)
        self.fcy1 = nn.Linear(latent_dim, n_traits)
        self.dropout = nn.Dropout(0.1)
        self.n_traits = n_traits

    def forward(self, x):
        z = F.relu(self.fc1(x))#,0.2)
        z = self.dropout(z)
        z = F.relu(self.fc2(z))#,0.2)
        z = self.dropout(z)
        z = F.relu(self.fc3(z))#,0.2)
        trait = self.fcy1(z)
        return trait.view(-1,self.n_traits)
    
    

