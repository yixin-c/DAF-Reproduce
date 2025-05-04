import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import random
import itertools

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

# UNet-style building blocks
class DownsampleBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, apply_norm=True):
        super(DownsampleBlock1D, self).__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=not apply_norm)
        self.norm = nn.InstanceNorm1d(out_channels) if apply_norm else nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class UpsampleBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=0.0):
        super(UpsampleBlock1D, self).__init__()
        
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x, skip=None):
        x = self.deconv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        if skip is not None:
            # Ensure x and skip have the same spatial dimension for concatenation
            if x.size(2) != skip.size(2):
                # Interpolate x to match skip's size
                x = F.interpolate(x, size=skip.size(2), mode='linear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            
        return x

class UNetGenerator1D(nn.Module):
    """UNet-style generator for 1D vectors with domain embedding"""
    def __init__(self, input_nc=1, output_nc=1, init_filters=16, n_domain_embedding=4, 
                 domains=2, log=True, dropout=0.0, embed_scale=0.1):
        super(UNetGenerator1D, self).__init__()
        
        # Input dimensions and latent dimensions
        self.vector_size = 2346  # Upper triangular part of 68x68 matrix
        self.n_domain_embedding = n_domain_embedding
        self.embed_scale = embed_scale  # Scaling factor for domain embedding
        
        # Debug flag
        self.debug = True
        
        # Encoder
        self.initial_pad = nn.ReflectionPad1d(3)
        self.initial_conv = nn.Conv1d(input_nc, init_filters, kernel_size=7, padding=0)
        self.initial_norm = nn.InstanceNorm1d(init_filters)
        self.initial_relu = nn.ReLU(inplace=True)
        
        # Downsampling layers - limit max filters to 128 to save memory
        filter_sizes = [
            init_filters, 
            min(init_filters*2, 64), 
            min(init_filters*4, 128), 
            min(init_filters*8, 128)
        ]
        
        self.down1 = DownsampleBlock1D(filter_sizes[0], filter_sizes[1])  # 2346 -> 1173
        self.down2 = DownsampleBlock1D(filter_sizes[1], filter_sizes[2])  # 1173 -> 587
        self.down3 = DownsampleBlock1D(filter_sizes[2], filter_sizes[3])  # 587 -> 293
        
        # We'll determine the actual latent length in the forward pass
        # Initialize with an estimate that will be updated
        self.latent_length = 293  # Updated from 294 to 293
        
        # Domain embedding - initialize with small weights
        self.embed = nn.Embedding(domains, 293 * n_domain_embedding)  # Using 293 instead of self.latent_length
        
        # Upsampling layers
        # Note: We concatenate skip connections, so input channels are doubled
        self.up1 = UpsampleBlock1D(filter_sizes[3] + n_domain_embedding, filter_sizes[2], dropout=dropout)  # 293 -> 587
        self.up2 = UpsampleBlock1D(filter_sizes[2] * 2, filter_sizes[1], dropout=dropout)  # 587 -> 1173
        self.up3 = UpsampleBlock1D(filter_sizes[1] * 2, filter_sizes[0], dropout=dropout)  # 1173 -> 2346
        
        # Final output layer
        self.final_pad = nn.ReflectionPad1d(3)
        self.final_conv = nn.Conv1d(filter_sizes[0] * 2, output_nc, kernel_size=7, padding=0)
        
        # Final activation - choose based on data preprocessing
        if log:
            self.final_activation = nn.ReLU()  # Linear activation for log-transformed data
        else:
            self.final_activation = nn.Tanh()  # Tanh for [-1, 1] range
    
    def forward(self, x, domain):
        if self.debug:
            print(f"Input shape: {x.shape}")
            
        # Encoding path
        x1 = self.initial_relu(self.initial_norm(self.initial_conv(self.initial_pad(x))))
        if self.debug:
            print(f"After initial conv: {x1.shape}")
            
        x2 = self.down1(x1)
        if self.debug:
            print(f"After down1: {x2.shape}")
            
        x3 = self.down2(x2)
        if self.debug:
            print(f"After down2: {x3.shape}")
            
        x4 = self.down3(x3)
        if self.debug:
            print(f"After down3 (bottleneck): {x4.shape}")
        
        # Update latent length to actual size
        actual_latent_length = x4.size(2)
        
        # Latent representation for dependence loss
        w = x4.view(x4.size(0), -1)  # Flattened latent representation
        if self.debug:
            print(f"Flattened latent (w): {w.shape}")
        
        # Domain embedding - reshape to match actual bottleneck size
        batch_size = x.size(0)
        domain_emb = self.embed(domain).view(batch_size, self.n_domain_embedding, -1)
        # Ensure domain embedding length matches bottleneck length
        if domain_emb.size(2) != actual_latent_length:
            domain_emb = F.interpolate(domain_emb, size=actual_latent_length, mode='linear', align_corners=False)
        domain_emb = domain_emb * self.embed_scale  # Scale embedding to prevent domination
        if self.debug:
            print(f"Domain embedding: {domain_emb.shape}")
        
        # Concatenate domain embedding with bottleneck features
        x4_with_domain = torch.cat([x4, domain_emb], dim=1)
        if self.debug:
            print(f"Bottleneck + domain: {x4_with_domain.shape}")
        
        # Decoding path with skip connections
        x = self.up1(x4_with_domain, x3)
        if self.debug:
            print(f"After up1 + skip: {x.shape}")
            
        x = self.up2(x, x2)
        if self.debug:
            print(f"After up2 + skip: {x.shape}")
            
        x = self.up3(x, x1)
        if self.debug:
            print(f"After up3 + skip: {x.shape}")
        
        # Final convolution and activation
        x = self.final_conv(self.final_pad(x))
        x = self.final_activation(x)
        if self.debug:
            print(f"Final output: {x.shape}")
        
        # Ensure output size matches input size
        if x.size(2) != self.vector_size:
            x = F.interpolate(x, size=self.vector_size, mode='linear', align_corners=False)
            if self.debug:
                print(f"After size correction: {x.shape}")
        
        # Disable debug after first run
        self.debug = False
        
        return x, w

class PatchDiscriminator1D(nn.Module):
    """PatchGAN discriminator for 1D data"""
    def __init__(self, input_nc=1, init_filters=16, n_layers=3):
        super(PatchDiscriminator1D, self).__init__()
        
        layers = [
            nn.Conv1d(input_nc, init_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        nf_mult = 1
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**i, 8)
            layers.extend([
                nn.Conv1d(init_filters * nf_mult_prev, init_filters * nf_mult, 
                          kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm1d(init_filters * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # Final layer
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers.extend([
            nn.Conv1d(init_filters * nf_mult_prev, init_filters * nf_mult, 
                      kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm1d(init_filters * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(init_filters * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ])
        
        self.model = nn.Sequential(*layers)
        
        # Add global average pooling to get a scalar output for each batch element
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        features = self.model(x)
        # Apply global pooling to get a scalar value
        output = self.global_pool(features)
        # Squeeze the last dimension to match target shape [batch_size, 1]
        return output.squeeze(-1) 
    