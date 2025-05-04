#!/usr/bin/env python
import sys
import torch
import random
import time
from models_unet import UNetGenerator1D, PatchDiscriminator1D, LambdaLR
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
import os
import logging
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import itertools
from argparse import Namespace
import pickle

args = Namespace(output_dir='./saved_model/ABCD_HCP_1D_UNet/',
                 device_id=3,
                 log=True,  # whether to apply log transformation to the adjacency matrices
                 nepoch=250,
                 train_bs=64,  # Reduced from 128 to 64
                 train_size=0.9,
                 dep_sigma=20,
                 init_filters=16,  # Reduced to match models_unet.py
                 n_domain_embedding=4,  # Reduced domain embedding channels
                 embed_scale=0.1,  # Scaling factor for domain embedding
                 dropout=0.0,
                 G_lr=0.0002,
                 DA_lr=0.0002,
                 DB_lr=0.0002,
                 evaluate_every=1,  # Frequency to monitor test loss
                 cyc_lambda=10.0,
                 cyc_BAB=True,
                 dep_lambda=1.,
                 dep_on='w')
log = args.log
sample = True

# Setup logging
os.makedirs('./logs', exist_ok=True)
logging.basicConfig(filename='./logs/daf-1d-unet-abcd-hcp.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

# Unsupervised DAF
## Load brain connectome data (adjacency matrices)
##################################################################################
dat_abcd = loadmat('../abcd/new_abcd_cog.mat')
dat_hcp = loadmat('../hcp/HCP_subcortical_CMData_desikan.mat')
tensor_abcd = dat_abcd['count_network']
tensor_hcp = dat_hcp['loaded_tensor_sub']

# Convert matrices to vectors (upper triangular part)
net_abcd = []
for i in range(tensor_abcd.shape[2]):
    ith = np.float32(tensor_abcd[:,:,i])
    if log:
        ith = np.log(ith+1)
    # Extract upper triangular part directly
    ith_vec = extract_upper(ith)
    net_abcd.append(ith_vec)
n_abcd = len(net_abcd)

net_hcp = []
for i in range(tensor_hcp.shape[3]):
    ith = np.float32(tensor_hcp[:,:,0,i] + np.transpose(tensor_hcp[:,:,0,i]))
    ith = ith[18:86, 18:86]  # Crop to 68x68 to match ABCD size
    if log:
        ith = np.log(ith+1)
    # Extract upper triangular part directly
    ith_vec = extract_upper(ith)
    net_hcp.append(ith_vec)
n_hcp = len(net_hcp)

## either sample or take all data from ABCD and HCP
if sample:
    _, id_abcd = train_test_split(range(n_abcd), test_size=892/n_abcd, random_state=42)
    _, id_hcp = train_test_split(range(n_hcp), test_size=173/n_hcp, random_state=42)
else:
    id_abcd = list(range(n_abcd))
    id_hcp = list(range(n_hcp))

net_abcd = [net_abcd[i] for i in id_abcd]
net_hcp = [net_hcp[i] for i in id_hcp]
##################################################################################
data_domain_0 = net_abcd
data_domain_1 = net_hcp

n_domain_0 = len(data_domain_0)
n_domain_1 = len(data_domain_1)
n_total = n_domain_0 + n_domain_1
print(f'Domain 0: {n_domain_0} subjects')
print(f'Domain 1: {n_domain_1} subjects')

# Combine the data
data_all = data_domain_0 + data_domain_1

## Preprocess the data
if args.log:
    # If log transformation was applied, stack the tensors
    data_all = torch.stack([torch.Tensor(i) for i in data_all]).view(-1, 1, 2346)  # 2346 = 68*(68+1)/2 (upper triangular elements)
else:
    # If no log transformation, scale data to range [0,1] (default in cycle gan)
    offset = np.max(data_all)/2
    data_all = torch.stack([torch.Tensor((i-offset)/offset) for i in data_all]).view(-1, 1, 2346)

# Domain labels:
labels = torch.cat((torch.tensor([0]).expand(n_domain_0, 1), torch.tensor([1]).expand(n_domain_1, 1)))

## Data loader
set_seed(1972014)
dataset = torch.utils.data.TensorDataset(data_all, labels)

# Split train and validation
train_id, test_id = train_test_split(list(range(n_total)), train_size=args.train_size, random_state=42)
train_dataset = torch.utils.data.Subset(dataset, train_id)
test_dataset = torch.utils.data.Subset(dataset, test_id)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_id), shuffle=False)

## DAF model configuration - using UNet generators
# Networks - using 1D UNet versions
netG_A2B = UNetGenerator1D(
    input_nc=1, 
    output_nc=1, 
    init_filters=args.init_filters,
    n_domain_embedding=args.n_domain_embedding,
    domains=2,
    log=args.log,
    dropout=args.dropout,
    embed_scale=args.embed_scale
).to(device)

netG_B2A = UNetGenerator1D(
    input_nc=1, 
    output_nc=1, 
    init_filters=args.init_filters,
    n_domain_embedding=args.n_domain_embedding,
    domains=2,
    log=args.log,
    dropout=args.dropout,
    embed_scale=args.embed_scale
).to(device)

netD_A = PatchDiscriminator1D(input_nc=1, init_filters=args.init_filters, n_layers=3).to(device)
netD_B = PatchDiscriminator1D(input_nc=1, init_filters=args.init_filters, n_layers=3).to(device)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                              lr=args.G_lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=args.DA_lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=args.DB_lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.nepoch, 0, args.nepoch/2).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(args.nepoch, 0, args.nepoch/2).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(args.nepoch, 0, args.nepoch/2).step)

## Train
# Use dictionaries to track losses and metrics
train_losses = {
    'G_total': [],        # Total generator loss
    'G_A2B': [],          # Generator A2B loss
    'G_B2A': [],          # Generator B2A loss
    'cycle_ABA': [],      # Cycle consistency ABA
    'cycle_BAB': [],      # Cycle consistency BAB
    'dep': [],            # Dependence loss
    'D_A': [],            # Discriminator A loss
    'D_B': [],            # Discriminator B loss
    'D_A_acc': [],        # Discriminator A accuracy
    'D_B_acc': [],        # Discriminator B accuracy
    'D_A_real_acc': [],   # Discriminator A accuracy on real samples
    'D_A_fake_acc': [],   # Discriminator A accuracy on fake samples
    'D_B_real_acc': [],   # Discriminator B accuracy on real samples
    'D_B_fake_acc': []    # Discriminator B accuracy on fake samples
}

test_losses = {
    'G_total': [],
    'G_A2B': [],
    'G_B2A': [],
    'cycle_ABA': [],
    'cycle_BAB': [],
    'dep': [],
    'D_A': [],
    'D_B': [],
    'D_A_acc': [],
    'D_B_acc': [],
    'D_A_real_acc': [],
    'D_A_fake_acc': [],
    'D_B_real_acc': [],
    'D_B_fake_acc': []
}

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.output_dir + 'figs/', exist_ok=True)

# Training loop
for epoch in range(0, args.nepoch):
    # Dictionary to store batch losses during this epoch
    batch_losses = {
        'G_total': [],
        'G_A2B': [],
        'G_B2A': [],
        'cycle_ABA': [],
        'cycle_BAB': [],
        'dep': [],
        'D_A': [],
        'D_B': [],
        'D_A_acc': [],
        'D_B_acc': [],
        'D_A_real_acc': [],
        'D_A_fake_acc': [],
        'D_B_real_acc': [],
        'D_B_fake_acc': []
    }

    tic = time.perf_counter()
    netG_A2B.train()
    netG_B2A.train()
    netD_A.train()
    netD_B.train()
    
    for i, (A, s) in enumerate(train_loader):
        # Set model input
        real_A = A.to(device)
        real_B = A.to(device)
        s = s.to(device)
        bs = real_A.shape[0]
        
        # Inputs & targets memory allocation
        Tensor = torch.FloatTensor
        target_real = Variable(Tensor(bs,1).fill_(1.0), requires_grad=False).to(device)
        target_fake = Variable(Tensor(bs,1).fill_(0.0), requires_grad=False).to(device)
        
        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()
        
        # GAN loss
        fake_B, w = netG_A2B(real_A, s)
        pred_fake_B = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake_B, target_real)
        
        fake_A, _ = netG_B2A(real_B, s)
        pred_fake_A = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake_A, target_real)
        
        # Cycle loss
        recovered_A, _ = netG_B2A(fake_B, s)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * args.cyc_lambda
        if args.cyc_BAB:
            recovered_B, _ = netG_A2B(fake_A, s)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * args.cyc_lambda
        else:
            loss_cycle_BAB = torch.tensor(0.)
        loss_cycle = loss_cycle_ABA + loss_cycle_BAB
        
        # Dependence loss - makes sure the transformed features are independent of domain 
        if args.dep_on == 'w':
            tmp = w.view(bs, -1)
        else:
            tmp = fake_B.view(bs, -1)
        s_view = s.view(bs, 1)
        DLR = args.dep_lambda * dependence(tmp, s_view, sig=args.dep_sigma, device=device) 

        # Total generator loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle + DLR
        loss_G.backward()
        optimizer_G.step()
        
        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real_A = netD_A(real_A)
        loss_D_real_A = criterion_GAN(pred_real_A, target_real)

        # Fake loss
        pred_fake_A = netD_A(fake_A.detach())
        loss_D_fake_A = criterion_GAN(pred_fake_A, target_fake)

        # Total loss
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
        loss_D_A.backward()
        optimizer_D_A.step()
        
        # Calculate Discriminator A accuracy - separated for real and fake
        acc_real_A = calculate_accuracy(pred_real_A, target_real)
        acc_fake_A = calculate_accuracy(pred_fake_A, target_fake)
        acc_D_A = (acc_real_A + acc_fake_A) / 2.0

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real_B = netD_B(real_B)
        loss_D_real_B = criterion_GAN(pred_real_B, target_real)
        
        # Fake loss
        pred_fake_B = netD_B(fake_B.detach())
        loss_D_fake_B = criterion_GAN(pred_fake_B, target_fake)

        # Total loss
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
        loss_D_B.backward()
        optimizer_D_B.step()
        
        # Calculate Discriminator B accuracy - separated for real and fake
        acc_real_B = calculate_accuracy(pred_real_B, target_real)
        acc_fake_B = calculate_accuracy(pred_fake_B, target_fake)
        acc_D_B = (acc_real_B + acc_fake_B) / 2.0

        # Store batch losses
        batch_losses['G_total'].append(loss_G.item())
        batch_losses['G_A2B'].append(loss_GAN_A2B.item())
        batch_losses['G_B2A'].append(loss_GAN_B2A.item())
        batch_losses['cycle_ABA'].append(loss_cycle_ABA.item())
        batch_losses['cycle_BAB'].append(loss_cycle_BAB.item())
        batch_losses['dep'].append(DLR.item())
        batch_losses['D_A'].append(loss_D_A.item())
        batch_losses['D_B'].append(loss_D_B.item())
        batch_losses['D_A_acc'].append(acc_D_A)
        batch_losses['D_B_acc'].append(acc_D_B)
        batch_losses['D_A_real_acc'].append(acc_real_A)
        batch_losses['D_A_fake_acc'].append(acc_fake_A)
        batch_losses['D_B_real_acc'].append(acc_real_B)
        batch_losses['D_B_fake_acc'].append(acc_fake_B)
        
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    
    # Calculate training metrics (average over batches)
    epoch_losses = {k: np.mean(v) for k, v in batch_losses.items()}
    
    toc = time.perf_counter()
    print(f"Trained epoch {epoch+1} in {toc - tic:0.4f} seconds") 
    
    # Evaluate model
    if (epoch + 1) % args.evaluate_every == 0:
        test_epoch_losses = evaluate_with_accuracy_dep(
            netG_A2B, netG_B2A, netD_A, netD_B, test_loader, epoch, 
            args.dep_sigma, args.cyc_lambda, args.dep_lambda, args.cyc_BAB, args.dep_on,
            device, save=True, plot=True, root=args.output_dir)
    
    # Log training results
    logging.info('-'*15+'train metrics'+'-'*15)    
    logging.info(f"Epoch: {epoch + 1}/{args.nepoch}, loss_G: {epoch_losses['G_total']:.3f}, loss_D_A: {epoch_losses['D_A']:.3f}, loss_D_B: {epoch_losses['D_B']:.3f}")
    logging.info(f"Epoch: {epoch + 1}/{args.nepoch}, G_A2B: {epoch_losses['G_A2B']:.3f}, G_B2A: {epoch_losses['G_B2A']:.3f}")
    logging.info(f"Epoch: {epoch + 1}/{args.nepoch}, cycle_ABA: {epoch_losses['cycle_ABA']:.3f}, cycle_BAB: {epoch_losses['cycle_BAB']:.3f}, dep: {epoch_losses['dep']:.3f}")
    logging.info(f"Epoch: {epoch + 1}/{args.nepoch}, D_A_acc: {epoch_losses['D_A_acc']:.3f} (real: {epoch_losses['D_A_real_acc']:.3f}, fake: {epoch_losses['D_A_fake_acc']:.3f})")
    logging.info(f"Epoch: {epoch + 1}/{args.nepoch}, D_B_acc: {epoch_losses['D_B_acc']:.3f} (real: {epoch_losses['D_B_real_acc']:.3f}, fake: {epoch_losses['D_B_fake_acc']:.3f})")
    logging.info('-'*15+' test metrics'+'-'*15)   
    logging.info(f"Epoch: {epoch + 1}/{args.nepoch}, loss_G: {test_epoch_losses['G_total']:.3f}, loss_D_A: {test_epoch_losses['D_A']:.3f}, loss_D_B: {test_epoch_losses['D_B']:.3f}")
    logging.info(f"Epoch: {epoch + 1}/{args.nepoch}, G_A2B: {test_epoch_losses['G_A2B']:.3f}, G_B2A: {test_epoch_losses['G_B2A']:.3f}")
    logging.info(f"Epoch: {epoch + 1}/{args.nepoch}, cycle_ABA: {test_epoch_losses['cycle_ABA']:.3f}, cycle_BAB: {test_epoch_losses['cycle_BAB']:.3f}, dep: {test_epoch_losses['dep']:.3f}")
    logging.info(f"Epoch: {epoch + 1}/{args.nepoch}, D_A_acc: {test_epoch_losses['D_A_acc']:.3f} (real: {test_epoch_losses['D_A_real_acc']:.3f}, fake: {test_epoch_losses['D_A_fake_acc']:.3f})")
    logging.info(f"Epoch: {epoch + 1}/{args.nepoch}, D_B_acc: {test_epoch_losses['D_B_acc']:.3f} (real: {test_epoch_losses['D_B_real_acc']:.3f}, fake: {test_epoch_losses['D_B_fake_acc']:.3f})")
    logging.info('*'*40)
    
    # Store metrics for plotting
    for k, v in epoch_losses.items():
        train_losses[k].append(v)
    
    for k, v in test_epoch_losses.items():
        test_losses[k].append(v)

# Save losses
with open(os.path.join(args.output_dir,'train_loss.pkl'), 'wb') as f:
    pickle.dump(train_losses, f)
with open(os.path.join(args.output_dir,'test_loss.pkl'), 'wb') as f:
    pickle.dump(test_losses, f)
        
# Plot loss curves
plt.figure(figsize=(18, 12))

# Plot main network losses
plt.subplot(3, 3, 1)
plt.plot(range(args.nepoch), train_losses['G_total'], label='Generator')
plt.plot(range(args.nepoch), train_losses['D_A'], label='Discriminator A')
plt.plot(range(args.nepoch), train_losses['D_B'], label='Discriminator B')
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3, 3, 2)
plt.plot(range(args.nepoch), test_losses['G_total'], label='Generator')
plt.plot(range(args.nepoch), test_losses['D_A'], label='Discriminator A')
plt.plot(range(args.nepoch), test_losses['D_B'], label='Discriminator B')
plt.title('Testing Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot discriminator accuracies
plt.subplot(3, 3, 4)
plt.plot(range(args.nepoch), train_losses['D_A_acc'], label='Overall')
plt.plot(range(args.nepoch), train_losses['D_A_real_acc'], label='Real')
plt.plot(range(args.nepoch), train_losses['D_A_fake_acc'], label='Fake')
plt.title('Train Discriminator A Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0, 1)

plt.subplot(3, 3, 5)
plt.plot(range(args.nepoch), test_losses['D_A_acc'], label='Overall')
plt.plot(range(args.nepoch), test_losses['D_A_real_acc'], label='Real')
plt.plot(range(args.nepoch), test_losses['D_A_fake_acc'], label='Fake')
plt.title('Test Discriminator A Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0, 1)

plt.subplot(3, 3, 7)
plt.plot(range(args.nepoch), train_losses['D_B_acc'], label='Overall')
plt.plot(range(args.nepoch), train_losses['D_B_real_acc'], label='Real')
plt.plot(range(args.nepoch), train_losses['D_B_fake_acc'], label='Fake')
plt.title('Train Discriminator B Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0, 1)

plt.subplot(3, 3, 8)
plt.plot(range(args.nepoch), test_losses['D_B_acc'], label='Overall')
plt.plot(range(args.nepoch), test_losses['D_B_real_acc'], label='Real')
plt.plot(range(args.nepoch), test_losses['D_B_fake_acc'], label='Fake')
plt.title('Test Discriminator B Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0, 1)

# Plot generator component losses
plt.subplot(3, 3, 3)
plt.plot(range(args.nepoch), train_losses['G_A2B'], label='G_A2B Loss')
plt.plot(range(args.nepoch), train_losses['G_B2A'], label='G_B2A Loss')
plt.plot(range(args.nepoch), train_losses['cycle_ABA'], label='Cycle ABA Loss')
plt.plot(range(args.nepoch), train_losses['cycle_BAB'], label='Cycle BAB Loss')
plt.plot(range(args.nepoch), train_losses['dep'], label='Dependence Loss')
plt.title('Training Generator Component Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3, 3, 6)
plt.plot(range(args.nepoch), test_losses['G_A2B'], label='G_A2B Loss')
plt.plot(range(args.nepoch), test_losses['G_B2A'], label='G_B2A Loss')
plt.plot(range(args.nepoch), test_losses['cycle_ABA'], label='Cycle ABA Loss')
plt.plot(range(args.nepoch), test_losses['cycle_BAB'], label='Cycle BAB Loss')
plt.plot(range(args.nepoch), test_losses['dep'], label='Dependence Loss')
plt.title('Testing Generator Component Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3, 3, 9)
# Add a text box summarizing the results
plt.axis('off')
plt.text(0.1, 0.9, f"Final metrics:", fontweight='bold')
plt.text(0.1, 0.8, f"Generator loss: {train_losses['G_total'][-1]:.4f} (train) / {test_losses['G_total'][-1]:.4f} (test)")
plt.text(0.1, 0.7, f"D_A accuracy: {train_losses['D_A_acc'][-1]:.4f} (train) / {test_losses['D_A_acc'][-1]:.4f} (test)")
plt.text(0.1, 0.6, f"  Real: {train_losses['D_A_real_acc'][-1]:.4f} (train) / {test_losses['D_A_real_acc'][-1]:.4f} (test)")
plt.text(0.1, 0.5, f"  Fake: {train_losses['D_A_fake_acc'][-1]:.4f} (train) / {test_losses['D_A_fake_acc'][-1]:.4f} (test)")
plt.text(0.1, 0.4, f"D_B accuracy: {train_losses['D_B_acc'][-1]:.4f} (train) / {test_losses['D_B_acc'][-1]:.4f} (test)")
plt.text(0.1, 0.3, f"  Real: {train_losses['D_B_real_acc'][-1]:.4f} (train) / {test_losses['D_B_real_acc'][-1]:.4f} (test)")
plt.text(0.1, 0.2, f"  Fake: {train_losses['D_B_fake_acc'][-1]:.4f} (train) / {test_losses['D_B_fake_acc'][-1]:.4f} (test)")

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'training_plots.png'))
plt.close() 