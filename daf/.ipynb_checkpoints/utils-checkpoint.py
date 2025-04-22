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
import os

    
def Gaussian_kernel(z, sig=2):
    # z.shape = (bs, dim)
    n = z.shape[0]
    d = z.shape[1]
    zi = z.repeat(1,n).view(n,n,d)
    zj = z.repeat(n,1).view(n,n,d)
    norm = torch.linalg.vector_norm(zi - zj, dim=2) 
    kz = - norm **2 / sig**2
    return torch.exp(kz)

def discrete_kernel(s):
    s = s.float()
    n = s.shape[0]
    si = s.repeat(1,n)
    sj = si.transpose(0,1)
    ks = si == sj
    return ks.float()

def dependence(z, s, sig=2, device='cpu'):
    size = z.shape[0]
    one_vec = torch.tensor([1.]).expand(size,1)
    H = torch.eye(size) - (1/size) * one_vec @ one_vec.transpose(0,1)
    H = H.to(device)
    ks = discrete_kernel(s)
    kz = Gaussian_kernel(z, sig = sig)
    Hz = H @ kz @ H
    Hs = H @ ks @ H
    DLR = torch.sum(Hz * Hs) / (size ** 2)
    return DLR

def img_grid_eval(epoch, x, xt, recov, s, file, nrow=1, ncol=4):
    # function to plot heatmap for original, transformed and reconstructed data (adjacency matrix)
    idx1 = (s==1).nonzero(as_tuple=True)[0]
    idx0 = (s==0).nonzero(as_tuple=True)[0]
    idx1 = idx1[random.sample(list(range(idx1.shape[0])),3)]
    idx0 = idx0[random.sample(list(range(idx0.shape[0])),3)]
    idx = torch.cat((idx0,idx1))
    if torch.is_tensor(x):
        x_imlist = [(item).permute(1,2,0) for item in x[idx]]
        xt_imlist = [(item).permute(1,2,0) for item in xt[idx]]
        recov_imlist = [(item).permute(1,2,0) for item in recov[idx]]
        
    else:
        x_imlist = [(item).transpose(1,2,0) for item in x[idx]]
        xt_imlist = [(item).transpose(1,2,0) for item in xt[idx]]
        recov_imlist = [(item).transpose(1,2,0) for item in recov[idx]]
    
    fig = plt.figure(figsize=(10., 5.), dpi = 200)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(nrow, ncol),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                    )
    imlist = x_imlist + xt_imlist + recov_imlist
    for ax, im in zip(grid, imlist):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
    # plt.suptitle('ff')
    plt.savefig(file)
    plt.close(fig)
    
def evaluate(netG_A2B, netG_B2A, netD_A, netD_B, loader, epoch, device, save=False, plot=False, root=None):
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    netG_A2B.eval()
    netG_B2A.eval()
    netD_A.eval()
    netD_B.eval()
    running_eval_loss = 0.0

    G_losses=[]
    DA_losses=[]
    DB_losses=[]
    gan=[]
    cyc=[]
    dep=[]
    with torch.no_grad():
        total=0
        
        for i, (A, s) in enumerate(loader):
        
            real_A = A.to(device)
            real_B = A.to(device)
            s = s.to(device)
            bs = real_A.shape[0]

            # Inputs & targets memory allocation
            Tensor = torch.FloatTensor
            target_real = Variable(Tensor(bs,1).fill_(1.0), requires_grad=False).to(device)
            target_fake = Variable(Tensor(bs,1).fill_(0.0), requires_grad=False).to(device)

            ###### Generators A2B and B2A ######

            # GAN loss 
            fake_B, _ = netG_A2B(real_A, s)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A, _ = netG_B2A(real_B, s)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A, _ = netG_B2A(fake_B, s)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

            recovered_B, _ = netG_A2B(fake_A, s)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

            # Dependence loss
            xt = fake_B.view(bs, -1)
            s = s.view(bs, 1)
            DLR = dependence(xt, s, sig=20, device=device)

            # Total loss
            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + DLR
            ###################################

            ###### Discriminator A ######
            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)
            # Fake loss
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)
            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            ###################################

            ###### Discriminator B ######
            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)
            # Fake loss
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)
            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            ###################################

            # collect loss
            G_losses.append(loss_G.data.item())
            DA_losses.append(loss_D_A.data.item())
            DB_losses.append(loss_D_B.data.item())
            gan.append(loss_GAN_A2B.data.item() + loss_GAN_B2A.data.item())
            cyc.append(loss_cycle_ABA.data.item() + loss_cycle_BAB.data.item())
            dep.append(DLR.data.item())
        G_loss_epoch =  np.mean(G_losses)
        DA_loss_epoch =  np.mean(DA_losses)
        DB_loss_epoch =  np.mean(DB_losses)
        gan_epoch =  np.mean(gan)
        cyc_epoch =  np.mean(cyc)
        dep_epoch =  np.mean(dep)
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

        x=x.detach().cpu()
        xt=xt.detach().cpu()
        recov=recov.detach().cpu()
        s=s.detach().cpu() 
        output_root = root+'figs/'
        if not os.path.exists(output_root):
            # Create the directory
            os.makedirs(output_root)
            print(f"Directory '{output_root}' was created.")
        img_grid_eval(epoch, x, xt, recov, s, file = output_root+'e'+str(epoch)+'.png',nrow=3, ncol=6)
    return G_loss_epoch, DA_loss_epoch, DB_loss_epoch, gan_epoch, cyc_epoch, dep_epoch



    
    
    
    

def img_grid(X, idx=range(4), nrow=1, ncol=4):
    if torch.is_tensor(X):
        imlist = [(x).permute(1,2,0) for x in X[idx]]
    else:
        imlist = [(x).transpose(1,2,0) for x in X[idx]]
    fig = plt.figure(figsize=(20., 80.), dpi = 200)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(nrow, ncol),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                    )
    
    for ax, im in zip(grid, imlist):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
    plt.show()
    