#!/usr/bin/env python
import sys
import torch
import random
import time
from models import *
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
        
        # # Output layer
        # if log:
        #     dc += [  nn.ReflectionPad2d(3),
        #              nn.Conv2d(init_depth, output_nc, 7),
        #              nn.ReLU()]
        # else:
        #     dc += [  nn.ReflectionPad2d(3),
        #              nn.Conv2d(init_depth, output_nc, 7),
        #              nn.Tanh() ]
        #### decode
        self.dc = nn.Sequential(*dc)
        
        # Dual-heads: mask and value
        self.mask_head = nn.Conv2d(in_features, 1, kernel_size=1)
        self.val_head  = nn.Conv2d(in_features, 1, kernel_size=1)
        
        self.log = log
        
    def forward(self, x, s):
        x = self.ec(x)
        x = self.z1(x)
        z = x.view(x.shape[0], -1)
        x = self.tconv2(self.tconv1(x))
        ##
        s_x = self.embed(s).view(x.shape[0], self.s_depth, self.z_size, self.z_size)
        xs = torch.cat((x,s_x),dim=1)
        
        # Decode backbone
        pre = self.dc(xs)
        # Heads
        mask_logit = self.mask_head(pre)
        val_log    = self.val_head(pre)
        # Probability mask
        P = torch.sigmoid(mask_logit)
        # Value to rate
        rate = torch.exp(val_log) if self.log else F.softplus(val_log)
        # Symmetrize mask and rate to preserve zeros
        P_sym   = torch.max(P, P.transpose(-1, -2))
        rate_sym= 0.5 * (rate + rate.transpose(-1, -2))
        # Combined adjacency
        Ahat = P_sym * rate_sym
        # Enforce symmetry and zero diag
        Ahat = 0.5 * (Ahat + Ahat.transpose(-1, -2))
        eye = torch.eye(Ahat.size(-1), device=Ahat.device)
        Ahat = Ahat * (1 - eye)
        return Ahat, z
    

# Setup logging
logging.basicConfig(filename='./logs/daf-sleep-abcd-modified.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed_value=973028):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy module
    torch.manual_seed(seed_value)  # PyTorch
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Python hash seed
    
    # CUDA reproducibility
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Unsupervised DAF

## Load brain connectome data with sleep information
print("Loading ABCD data with sleep information...")
dat_abcd = loadmat('../abcd/new_abcd_cog_with_sleep_filtered.mat')
tensor_abcd = dat_abcd['count_network']
sleep_domain_numeric = dat_abcd['sleep_domain_numeric']

# whether to apply log transformation to the adjacency matrices
log = True

set_seed(1972014)

# Extract matrices for short sleep (0) and long sleep (1)
net_short_sleep = []
net_long_sleep = []

for i in range(tensor_abcd.shape[2]):
    ith = np.float32(tensor_abcd[:,:,i])
    if log:
        ith = np.log(ith+1)
    ith = ith.reshape(68*68)
    
    # Check sleep domain (0=short, 1=long)
    sleep_value = sleep_domain_numeric[i][0]
    if sleep_value == 0:  # Short sleep
        net_short_sleep.append(ith)
    elif sleep_value == 1:  # Long sleep
        net_long_sleep.append(ith)

n_short = len(net_short_sleep)
n_long = len(net_long_sleep)

print(f'Short sleep: {n_short} subjects')
print(f'Long sleep: {n_long} subjects')

# Combine the data
net_all = net_short_sleep + net_long_sleep

## Preprocess the data
if log:
    # If log transformation was applied, stack the tensors
    net_all = torch.stack([torch.Tensor(i) for i in net_all]).view(-1, 1, 68, 68)
else:
    # If no log transformation, scale data to range [0,1] (default in cycle gan)
    offset = np.max(net_all)/2
    net_all = torch.stack([torch.Tensor((i-offset)/offset) for i in net_all]).view(-1, 1, 68, 68)

# Domain labels: Short Sleep (0), Long Sleep (1)
labels = torch.cat((torch.tensor([0]).expand(n_short, 1), torch.tensor([1]).expand(n_long, 1)))

## Data loader
set_seed(1972034)
nepoch = 300
dataset = utils.TensorDataset(net_all, labels)

# Split train and validation
train_id, test_id = train_test_split(list(range(n_short + n_long)), train_size=0.9, random_state=42)
train_dataset = utils.Subset(dataset, train_id)
test_dataset = utils.Subset(dataset, test_id)
train_loader = utils.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = utils.DataLoader(test_dataset, batch_size=len(test_id), shuffle=False)

## DAF model configuration
# Networks
set_seed(1972014)
netG_A2B = Generator(input_nc=1, output_nc=1, init_depth=48, s_depth=10, n_residual_blocks=5).to(device)
netG_B2A = Generator(input_nc=1, output_nc=1, init_depth=48, s_depth=10, n_residual_blocks=5).to(device)
netD_A = Discriminator(input_nc=1, init_depth=48).to(device)
netD_B = Discriminator(input_nc=1, init_depth=48).to(device)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(nepoch, 0, nepoch/2).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(nepoch, 0, nepoch/2).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(nepoch, 0, nepoch/2).step)

## Train
train_G_loss = []
test_G_loss = []
train_DA_loss = []
test_DA_loss = []
train_DB_loss = []
test_DB_loss = []

# New metrics for tracking discriminator accuracy
train_DA_acc = []
test_DA_acc = []
train_DB_acc = []
test_DB_acc = []

# Track component losses
train_gan_loss = []
test_gan_loss = []
train_cyc_loss = []
test_cyc_loss = []
train_dep_loss = []
test_dep_loss = []

# Frequency to monitor test loss
evaluate_every = 1

# Function to calculate discriminator accuracy
def calculate_accuracy(pred, target, threshold=0.5):
    # Apply sigmoid to convert raw scores to probabilities
    pred_prob = torch.sigmoid(pred)
    # Convert probabilities to binary predictions
    pred_binary = (pred_prob > threshold).float()
    correct = (pred_binary == target).float().sum()
    accuracy = correct / target.size(0)
    return accuracy.item()

# Modified evaluate function to track discriminator accuracy
def evaluate_with_accuracy(netG_A2B, netG_B2A, netD_A, netD_B, loader, epoch, device, save=False, plot=False, root=None):
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    netG_A2B.eval()
    netG_B2A.eval()
    netD_A.eval()
    netD_B.eval()

    G_losses = []
    DA_losses = []
    DB_losses = []
    DA_accs = []
    DB_accs = []
    gan = []
    cyc = []
    dep = []

    with torch.no_grad():
        for i, (A, s) in enumerate(loader):
            real_A = A.to(device)
            real_B = A.to(device)
            s = s.to(device)
            bs = real_A.shape[0]

            # Inputs & targets memory allocation
            Tensor = torch.FloatTensor
            target_real = Variable(Tensor(bs, 1).fill_(1.0), requires_grad=False).to(device)
            target_fake = Variable(Tensor(bs, 1).fill_(0.0), requires_grad=False).to(device)

            # Generator evaluation
            fake_B, _ = netG_A2B(real_A, s)
            pred_fake_B = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake_B, target_real)

            fake_A, _ = netG_B2A(real_B, s)
            pred_fake_A = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake_A, target_real)

            # Cycle loss
            recovered_A, _ = netG_B2A(fake_B, s)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

            recovered_B, _ = netG_A2B(fake_A, s)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

            # Dependence loss
            xt = fake_B.view(bs, -1)
            s_view = s.view(bs, 1)
            DLR = dependence(xt, s_view, sig=10000, device=device)

            # Total generator loss
            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + DLR

            # Discriminator A evaluation
            pred_real_A = netD_A(real_A)
            loss_D_real_A = criterion_GAN(pred_real_A, target_real)
            pred_fake_A = netD_A(fake_A.detach())
            loss_D_fake_A = criterion_GAN(pred_fake_A, target_fake)
            loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5

            # Discriminator B evaluation
            pred_real_B = netD_B(real_B)
            loss_D_real_B = criterion_GAN(pred_real_B, target_real)
            pred_fake_B = netD_B(fake_B.detach())
            loss_D_fake_B = criterion_GAN(pred_fake_B, target_fake)
            loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5

            # Calculate discriminator accuracies
            acc_real_A = calculate_accuracy(pred_real_A, target_real)
            acc_fake_A = calculate_accuracy(pred_fake_A, target_fake)
            acc_D_A = (acc_real_A + acc_fake_A) / 2.0

            acc_real_B = calculate_accuracy(pred_real_B, target_real)
            acc_fake_B = calculate_accuracy(pred_fake_B, target_fake)
            acc_D_B = (acc_real_B + acc_fake_B) / 2.0

            # Collect metrics
            G_losses.append(loss_G.data.item())
            DA_losses.append(loss_D_A.data.item())
            DB_losses.append(loss_D_B.data.item())
            DA_accs.append(acc_D_A)
            DB_accs.append(acc_D_B)
            gan.append(loss_GAN_A2B.data.item() + loss_GAN_B2A.data.item())
            cyc.append(loss_cycle_ABA.data.item() + loss_cycle_BAB.data.item())
            dep.append(DLR.data.item())

        # Calculate mean values
        G_loss_epoch = np.mean(G_losses)
        DA_loss_epoch = np.mean(DA_losses)
        DB_loss_epoch = np.mean(DB_losses)
        DA_acc_epoch = np.mean(DA_accs)
        DB_acc_epoch = np.mean(DB_accs)
        gan_epoch = np.mean(gan)
        cyc_epoch = np.mean(cyc)
        dep_epoch = np.mean(dep)

    if save:
        if not os.path.exists(root):
            # Create the directory
            os.makedirs(root)
            print(f"Directory '{root}' was created.")
            
        torch.save(netG_A2B.state_dict(), root+'G_A2B_e'+str(epoch)+'_G'+str(np.round(G_loss_epoch,3))\
                   +'_gan'+str(np.round(gan_epoch,3)) + '_cyc'+str(np.round(cyc_epoch)) + '_dep'+ str(np.round(dep_epoch*100,3)) + '.pth')
        torch.save(netG_B2A.state_dict(), root+'G_B2A_e'+str(epoch)+'_G'+str(np.round(G_loss_epoch,3))\
                   +'_gan'+str(np.round(gan_epoch,3)) + '_cyc'+str(np.round(cyc_epoch)) + '_dep'+ str(np.round(dep_epoch*100,3)) + '.pth')
        torch.save(netD_A.state_dict(), root+'D_A_e'+str(epoch)+'_'+str(np.round(DA_loss_epoch,3))+'.pth')
        torch.save(netD_B.state_dict(), root+'D_B_e'+str(epoch)+'_'+str(np.round(DB_loss_epoch,3))+'.pth')
        
    if plot:
        with torch.no_grad():
            for i, (x, s) in enumerate(loader):
                x, s = x.to(device), s.to(device)
                xt, _ = netG_A2B(x, s)
                recov, _ = netG_B2A(xt, s)

        x = x.detach().cpu()
        xt = xt.detach().cpu()
        recov = recov.detach().cpu()
        s = s.detach().cpu() 
        output_root = root+'figs/'
        if not os.path.exists(output_root):
            # Create the directory
            os.makedirs(output_root)
            print(f"Directory '{output_root}' was created.")
        img_grid_eval(epoch, x, xt, recov, s, file = output_root+'e'+str(epoch)+'.png', nrow=3, ncol=6)
    
    return G_loss_epoch, DA_loss_epoch, DB_loss_epoch, DA_acc_epoch, DB_acc_epoch, gan_epoch, cyc_epoch, dep_epoch

# Training loop
for epoch in range(0, nepoch):
    G_losses = []
    DA_losses = []
    DB_losses = []
    DA_accs = []
    DB_accs = []
    gan = []
    cyc = []
    dep = []

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
        fake_B, _ = netG_A2B(real_A, s)
        pred_fake_B = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake_B, target_real)
        
        fake_A, _ = netG_B2A(real_B, s)
        pred_fake_A = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake_A, target_real)
        
        # Cycle loss
        recovered_A, _ = netG_B2A(fake_B, s)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B, _ = netG_A2B(fake_A, s)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0
        
        # Dependence loss - makes sure the transformed features are independent of domain 
        xt = fake_B.view(bs, -1)
        s_view = s.view(bs, 1)
        DLR = dependence(xt, s_view, sig=10000, device=device) 

        # Total generator loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + DLR
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
        
        # Calculate Discriminator A accuracy
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
        
        # Calculate Discriminator B accuracy
        acc_real_B = calculate_accuracy(pred_real_B, target_real)
        acc_fake_B = calculate_accuracy(pred_fake_B, target_fake)
        acc_D_B = (acc_real_B + acc_fake_B) / 2.0
        
        # Collect metrics
        G_losses.append(loss_G.data.item())
        DA_losses.append(loss_D_A.data.item())
        DB_losses.append(loss_D_B.data.item())
        DA_accs.append(acc_D_A)
        DB_accs.append(acc_D_B)
        gan.append(loss_GAN_A2B.data.item() + loss_GAN_B2A.data.item())
        cyc.append(loss_cycle_ABA.data.item() + loss_cycle_BAB.data.item())
        dep.append(DLR.data.item())
        
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    
    # Calculate training metrics
    G_loss_epoch = torch.mean(torch.FloatTensor(G_losses))
    DA_loss_epoch = torch.mean(torch.FloatTensor(DA_losses))
    DB_loss_epoch = torch.mean(torch.FloatTensor(DB_losses))
    DA_acc_epoch = np.mean(DA_accs)
    DB_acc_epoch = np.mean(DB_accs)
    gan_epoch = torch.mean(torch.FloatTensor(gan))
    cyc_epoch = torch.mean(torch.FloatTensor(cyc))
    dep_epoch = torch.mean(torch.FloatTensor(dep))
    
    toc = time.perf_counter()
    print(f"Trained epoch {epoch+1} in {toc - tic:0.4f} seconds") 
    
    # Evaluate model
    if (epoch + 1) % evaluate_every == 0:
        G_loss_test, DA_loss_test, DB_loss_test, DA_acc_test, DB_acc_test, gan_test, cyc_test, dep_test = evaluate_with_accuracy(
            netG_A2B, netG_B2A, netD_A, netD_B, test_loader, epoch, device, 
            save=True, plot=True, root='./saved_model/sleep_daf/')
    
    # Log and print training results
    logging.info('-'*15+'train metrics'+'-'*15)    
    logging.info(f"Epoch: {epoch + 1}/{nepoch}, loss_D_A: {DA_loss_epoch:.3f}, loss_D_B: {DB_loss_epoch:.3f}, loss_G: {G_loss_epoch:.3f}")
    logging.info(f"Epoch: {epoch + 1}/{nepoch}, acc_D_A: {DA_acc_epoch:.3f}, acc_D_B: {DB_acc_epoch:.3f}")
    logging.info(f"Epoch: {epoch + 1}/{nepoch}, loss_gan: {gan_epoch:.3f}, loss_cyc: {cyc_epoch:.3f}, loss_dep: {dep_epoch:.3f}")
    logging.info('-'*15+' test metrics'+'-'*15)   
    logging.info(f"Epoch: {epoch + 1}/{nepoch}, loss_D_A: {DA_loss_test:.3f}, loss_D_B: {DB_loss_test:.3f}, loss_G: {G_loss_test:.3f}")
    logging.info(f"Epoch: {epoch + 1}/{nepoch}, acc_D_A: {DA_acc_test:.3f}, acc_D_B: {DB_acc_test:.3f}")
    logging.info(f"Epoch: {epoch + 1}/{nepoch}, loss_gan: {gan_test:.3f}, loss_cyc: {cyc_test:.3f}, loss_dep: {dep_test:.3f}")
    logging.info('*'*40)

    print('-'*15+'train metrics'+'-'*15)    
    print(f"Epoch: {epoch + 1}/{nepoch}, loss_D_A: {DA_loss_epoch:.3f}, loss_D_B: {DB_loss_epoch:.3f}, loss_G: {G_loss_epoch:.3f}")
    print(f"Epoch: {epoch + 1}/{nepoch}, acc_D_A: {DA_acc_epoch:.3f}, acc_D_B: {DB_acc_epoch:.3f}")
    print(f"Epoch: {epoch + 1}/{nepoch}, loss_gan: {gan_epoch:.3f}, loss_cyc: {cyc_epoch:.3f}, loss_dep: {dep_epoch:.3f}")
    print('-'*15+' test metrics'+'-'*15)   
    print(f"Epoch: {epoch + 1}/{nepoch}, loss_D_A: {DA_loss_test:.3f}, loss_D_B: {DB_loss_test:.3f}, loss_G: {G_loss_test:.3f}")
    print(f"Epoch: {epoch + 1}/{nepoch}, acc_D_A: {DA_acc_test:.3f}, acc_D_B: {DB_acc_test:.3f}")
    print(f"Epoch: {epoch + 1}/{nepoch}, loss_gan: {gan_test:.3f}, loss_cyc: {cyc_test:.3f}, loss_dep: {dep_test:.3f}")
    print('*'*40)
    
    # Store metrics for plotting
    train_G_loss.append(G_loss_epoch.item())
    test_G_loss.append(G_loss_test)
    train_DA_loss.append(DA_loss_epoch.item())
    test_DA_loss.append(DA_loss_test)
    train_DB_loss.append(DB_loss_epoch.item())
    test_DB_loss.append(DB_loss_test)
    train_DA_acc.append(DA_acc_epoch)
    test_DA_acc.append(DA_acc_test)
    train_DB_acc.append(DB_acc_epoch)
    test_DB_acc.append(DB_acc_test)
    # Store component losses
    train_gan_loss.append(gan_epoch.item())
    test_gan_loss.append(gan_test)
    train_cyc_loss.append(cyc_epoch.item())
    test_cyc_loss.append(cyc_test)
    train_dep_loss.append(dep_epoch.item())
    test_dep_loss.append(dep_test)

# Plot loss curves
plt.figure(figsize=(18, 12))
plt.subplot(3, 2, 1)
plt.plot(range(nepoch), train_G_loss, label='Train G')
plt.plot(range(nepoch), train_DA_loss, label='Train D_A')
plt.plot(range(nepoch), train_DB_loss, label='Train D_B')
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(range(nepoch), test_G_loss, label='Test G')
plt.plot(range(nepoch), test_DA_loss, label='Test D_A')
plt.plot(range(nepoch), test_DB_loss, label='Test D_B')
plt.title('Testing Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy curves
plt.subplot(3, 2, 3)
plt.plot(range(nepoch), train_DA_acc, label='Train D_A')
plt.plot(range(nepoch), train_DB_acc, label='Train D_B')
plt.title('Training Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0, 1)

plt.subplot(3, 2, 4)
plt.plot(range(nepoch), test_DA_acc, label='Test D_A')
plt.plot(range(nepoch), test_DB_acc, label='Test D_B')
plt.title('Testing Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0, 1)

# Plot component losses
plt.subplot(3, 2, 5)
plt.plot(range(nepoch), train_gan_loss, label='GAN Loss')
plt.plot(range(nepoch), train_cyc_loss, label='Cycle Loss')
plt.plot(range(nepoch), train_dep_loss, label='Dependence Loss')
plt.title('Training Component Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(range(nepoch), test_gan_loss, label='GAN Loss')
plt.plot(range(nepoch), test_cyc_loss, label='Cycle Loss')
plt.plot(range(nepoch), test_dep_loss, label='Dependence Loss')
plt.title('Testing Component Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('./saved_model/sleep_daf/training_plots.png')
plt.close()

print("Training complete. Results saved to ./saved_model/sleep_daf/") 