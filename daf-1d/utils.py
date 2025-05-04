import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import math
from scipy.io import savemat

def set_seed(seed_value=973028):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy module
    torch.manual_seed(seed_value)  # PyTorch
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Python hash seed
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8" for newer CUDA
    
    # CUDA reproducibility
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Matrix-Vector conversion utilities
def extract_upper(matrices, output_type='auto'):
    # Extract upper triangular part of matrices
    is_tensor = isinstance(matrices, torch.Tensor)
    arr = matrices if is_tensor else np.array(matrices)
    orig_ndim = arr.ndim
    if arr.ndim == 2:
        arr = arr[None, None, ...]
    elif arr.ndim == 3:
        arr = arr[:, None, ...]
    elif arr.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported input ndim={arr.ndim}")
    B, C, V, _ = arr.shape
    assert V == arr.shape[-1]
    if is_tensor:
        idx = torch.triu_indices(V, V)
        vec = arr[:, :, idx[0], idx[1]]
    else:
        idx = np.triu_indices(V)
        vec = arr[:, :, idx[0], idx[1]]
    if orig_ndim == 2:
        vec = vec[0,0]
    elif orig_ndim == 3:
        vec = vec[:,0]
    if output_type == 'numpy':
        if is_tensor:
            vec = vec.cpu().numpy()
    elif output_type == 'torch':
        if not is_tensor:
            vec = torch.from_numpy(vec)
    elif output_type != 'auto':
        raise ValueError("output_type must be 'auto','numpy', or 'torch'")
    return vec

def reconstruct_symmetric(vectors, output_type='auto'):
    # Reconstruct symmetric matrix from upper triangular vectors
    is_tensor = isinstance(vectors, torch.Tensor)
    vecs = vectors if is_tensor else np.array(vectors)
    orig_ndim = vecs.ndim
    if vecs.ndim == 1:
        vecs = vecs[None, None, :]
    elif vecs.ndim == 2:
        vecs = vecs[:, None, :]
    elif vecs.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported input ndim={vecs.ndim}")
    B, C, K = vecs.shape

    # infer V
    V = int((math.sqrt(8*K + 1) - 1) / 2)
    if V*(V+1)//2 != K:
        raise ValueError(f"Vector length {K} is not n(n+1)/2 for any integer n.")

    if is_tensor:
        # choose a float dtype if input was integer
        tensor_dtype = vecs.dtype if vecs.dtype.is_floating_point else torch.float32
        idx = torch.triu_indices(V, V)
        mats = torch.zeros((B, C, V, V), device=vecs.device, dtype=tensor_dtype)
        mats[:, :, idx[0], idx[1]] = vecs.to(tensor_dtype)
        mats = mats + mats.transpose(2,3)
        d = torch.arange(V, device=vecs.device)
        mats[:, :, d, d] /= 2
    else:
        # choose float64 if input was integer
        array_dtype = vecs.dtype if np.issubdtype(vecs.dtype, np.floating) else np.float64
        idx = np.triu_indices(V)
        mats = np.zeros((B, C, V, V), dtype=array_dtype)
        mats[:, :, idx[0], idx[1]] = vecs.astype(array_dtype)
        mats = mats + mats.transpose(0,1,3,2)
        for b in range(B):
            for c in range(C):
                mats[b, c, :, :].flat[idx[0]*V + idx[1]]  # no-op to ensure view
                # divide diagonal
                for d in range(V):
                    mats[b, c, d, d] /= 2

    if orig_ndim == 1:
        mats = mats[0,0]
    elif orig_ndim == 2:
        mats = mats[:,0]
    if output_type == 'numpy':
        if is_tensor:
            mats = mats.cpu().numpy()
    elif output_type == 'torch':
        if not is_tensor:
            mats = torch.from_numpy(mats)
    elif output_type != 'auto':
        raise ValueError("output_type must be 'auto','numpy', or 'torch'")

    return mats

def Gaussian_kernel(z, sig=2):
    # z.shape = (bs, dim)
    n = z.shape[0]
    d = z.shape[1]
    zi = z.repeat(1, n).view(n, n, d)
    zj = z.repeat(n, 1).view(n, n, d)
    norm = torch.linalg.vector_norm(zi - zj, dim=2) 
    kz = - norm ** 2 / sig**2
    return torch.exp(kz)

def discrete_kernel(s):
    s = s.float()
    n = s.shape[0]
    si = s.repeat(1, n)
    sj = si.transpose(0, 1)
    ks = si == sj
    return ks.float()

def dependence(z, s, sig=2, device='cpu'):
    size = z.shape[0]
    one_vec = torch.tensor([1.]).expand(size, 1)
    H = torch.eye(size) - (1/size) * one_vec @ one_vec.transpose(0, 1)
    H = H.to(device)
    ks = discrete_kernel(s)
    kz = Gaussian_kernel(z, sig=sig)
    Hz = H @ kz @ H
    Hs = H @ ks @ H
    DLR = torch.sum(Hz * Hs) / (size ** 2)
    return DLR

def img_grid_eval(epoch, x, xt, recov, s, file, nrow=1, ncol=4):
    # Function to plot heatmap for original, transformed and reconstructed data
    # Convert 1D vectors back to 2D matrices for visualization
    idx1 = (s==1).nonzero(as_tuple=True)[0]
    idx0 = (s==0).nonzero(as_tuple=True)[0]
    idx1 = idx1[random.sample(list(range(idx1.shape[0])), min(3, idx1.shape[0]))]
    idx0 = idx0[random.sample(list(range(idx0.shape[0])), min(3, idx0.shape[0]))]
    idx = torch.cat((idx0, idx1))
    
    # Convert 1D vectors to 2D matrices for visualization
    if torch.is_tensor(x):
        x_matrices = reconstruct_symmetric(x[idx].squeeze(1))
        xt_matrices = reconstruct_symmetric(xt[idx].squeeze(1))
        recov_matrices = reconstruct_symmetric(recov[idx].squeeze(1))
        
        x_imlist = [(item).permute(1, 0) for item in x_matrices]
        xt_imlist = [(item).permute(1, 0) for item in xt_matrices]
        recov_imlist = [(item).permute(1, 0) for item in recov_matrices]
    else:
        x_matrices = reconstruct_symmetric(x[idx][:, 0, :])
        xt_matrices = reconstruct_symmetric(xt[idx][:, 0, :])
        recov_matrices = reconstruct_symmetric(recov[idx][:, 0, :])
        
        x_imlist = [item for item in x_matrices]
        xt_imlist = [item for item in xt_matrices]
        recov_imlist = [item for item in recov_matrices]
    
    fig = plt.figure(figsize=(10., 5.), dpi=200)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(nrow, ncol),
                     axes_pad=0.1)
    
    imlist = x_imlist + xt_imlist + recov_imlist
    for ax, im in zip(grid, imlist):
        ax.imshow(im)
    
    plt.savefig(file)
    plt.close(fig)

# Function to calculate discriminator accuracy
def calculate_accuracy(pred, target, threshold=0.5, sigmoid=False):
    # Apply sigmoid to convert raw scores to probabilities
    if sigmoid:
        pred_prob = torch.sigmoid(pred)
    else:
        pred_prob = pred
    # Convert probabilities to binary predictions
    pred_binary = (pred_prob > threshold).float()
    correct = (pred_binary == target).float().sum()
    accuracy = correct / target.size(0)
    return accuracy.item()

# Modified evaluate function for 1D models
def evaluate_with_accuracy_dep(netG_A2B, netG_B2A, netD_A, netD_B, loader, epoch, dep_sigma, 
                             cyc_lambda, dep_lambda, cyc_BAB, dep_on, device, save=False, 
                             plot=False, root=None):
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    netG_A2B.eval()
    netG_B2A.eval()
    netD_A.eval()
    netD_B.eval()

    # Batch losses dictionary
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
            fake_B, w = netG_A2B(real_A, s)
            pred_fake_B = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake_B, target_real)

            fake_A, _ = netG_B2A(real_B, s)
            pred_fake_A = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake_A, target_real)

            # Cycle loss
            recovered_A, _ = netG_B2A(fake_B, s)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * cyc_lambda
            if cyc_BAB:
                recovered_B, _ = netG_A2B(fake_A, s)
                loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * cyc_lambda
            else:
                loss_cycle_BAB = torch.tensor(0.)
            loss_cycle = loss_cycle_ABA + loss_cycle_BAB

            # Dependence loss
            if dep_on == 'w':
                tmp = w.view(bs, -1)
            else:
                tmp = fake_B.view(bs, -1)
            s_view = s.view(bs, 1)
            DLR = dep_lambda * dependence(tmp, s_view, sig=dep_sigma, device=device)

            # Total generator loss
            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle + DLR

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

        # Calculate mean values
        epoch_losses = {k: np.mean(v) for k, v in batch_losses.items()}

    if save:
        if not os.path.exists(root):
            os.makedirs(root)
            print(f"Directory '{root}' was created.")
            
        # Save model checkpoints with key metrics in filename
        torch.save(netG_A2B.state_dict(), 
                  f"{root}G_A2B_e{epoch}_G{epoch_losses['G_total']:.3f}_gan{epoch_losses['G_A2B'] + epoch_losses['G_B2A']:.3f}_cyc{epoch_losses['cycle_ABA'] + epoch_losses['cycle_BAB']:.3f}_dep{epoch_losses['dep']*100:.3f}.pth")
        torch.save(netG_B2A.state_dict(), 
                  f"{root}G_B2A_e{epoch}_G{epoch_losses['G_total']:.3f}_gan{epoch_losses['G_A2B'] + epoch_losses['G_B2A']:.3f}_cyc{epoch_losses['cycle_ABA'] + epoch_losses['cycle_BAB']:.3f}_dep{epoch_losses['dep']*100:.3f}.pth")
        torch.save(netD_A.state_dict(), f"{root}D_A_e{epoch}_{epoch_losses['D_A']:.3f}.pth")
        torch.save(netD_B.state_dict(), f"{root}D_B_e{epoch}_{epoch_losses['D_B']:.3f}.pth")
        
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
            os.makedirs(output_root)
            print(f"Directory '{output_root}' was created.")
        img_grid_eval(epoch, x, xt, recov, s, file=output_root+f'e{epoch}.png', nrow=3, ncol=6)
    
    return epoch_losses

def img_grid(X, idx=range(4), nrow=1, ncol=4):
    # Convert 1D vectors to 2D matrices for visualization
    if torch.is_tensor(X):
        X_matrices = reconstruct_symmetric(X[idx].squeeze(1))
        imlist = [(x).permute(1, 0) for x in X_matrices]
    else:
        X_matrices = reconstruct_symmetric(X[idx][:, 0, :])
        imlist = [x for x in X_matrices]
    
    fig = plt.figure(figsize=(20., 80.), dpi=200)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(nrow, ncol),
                     axes_pad=0.1)
    
    for ax, im in zip(grid, imlist):
        ax.imshow(im)
    plt.show()

def extract_results_1d(data_loader, netG_A2B, netG_B2A, set_name, device, log=True, offset=None, data_all=None, save=False, save_path=None):
    """
    Extract results from the trained models for the given data loader.
    Keeps data in vector form (upper triangular part of matrices).
    
    Args:
        data_loader: PyTorch data loader
        netG_A2B: Generator A to B
        netG_B2A: Generator B to A
        set_name: String identifier for the dataset (e.g., 'train', 'test')
        device: Device to run computations on
        log: Whether log transform was applied to the data
        offset: Offset value used for normalization (if applied)
        data_all: All data for reference if needed for normalization
        save: Whether to save results
        save_path: Path to save results
        
    Returns:
        Dictionary containing results for both domains, with data in vector form
    """
    from scipy.io import savemat
    import os
    
    # Initialize lists to store all data
    all_X = []
    all_A = []
    all_R = []
    all_w = []
    all_z = []
    all_labels = []
    
    print(f"Processing {set_name} dataset...")
    with torch.no_grad():
        for batch_idx, (X, s) in enumerate(data_loader):
            # Print progress
            print(f"Processing batch {batch_idx+1}/{len(data_loader)}")

            # Move data to device
            X = X.to(device)
            s = s.to(device)

            # Forward pass through generators
            A, w = netG_A2B(X, s)  # A is harmonized data, w is embedding of observed data
            R, z = netG_B2A(A, s)  # R is reconstructed data, z is domain invariant embedding

            # Get domain labels
            s_cpu = s.cpu().numpy().flatten()

            # Detach and move to CPU
            X_cpu = X.cpu().numpy()
            A_cpu = A.cpu().numpy()
            w_cpu = w.cpu().numpy()
            R_cpu = R.cpu().numpy()
            z_cpu = z.cpu().numpy()

            # Keep data in vector form (don't convert to matrices)
            # Just apply inverse transformations if needed
            if log:
                # Inverse log transform for visualization: exp(x) - 1
                X_orig = np.exp(X_cpu) - 1
                A_orig = np.exp(A_cpu) - 1
                R_orig = np.exp(R_cpu) - 1
            else:
                # Return to original scale if offset was used
                if offset is None and data_all is not None:
                    # Calculate offset from data if not provided
                    if isinstance(data_all, torch.Tensor):
                        data_all_np = data_all.cpu().numpy()
                    else:
                        data_all_np = data_all
                    offset = np.max(data_all_np)/2
                
                if offset is not None:
                    X_orig = X_cpu * offset + offset
                    A_orig = A_cpu * offset + offset
                    R_orig = R_cpu * offset + offset
                else:
                    # If no scaling was done
                    X_orig = X_cpu
                    A_orig = A_cpu
                    R_orig = R_cpu

            # Collect data from this batch
            all_X.append(X_orig)
            all_A.append(A_orig)
            all_R.append(R_orig)
            all_w.append(w_cpu)
            all_z.append(z_cpu)
            all_labels.append(s_cpu)

        # Concatenate all batches
        all_X = np.concatenate(all_X, axis=0)
        all_A = np.concatenate(all_A, axis=0)
        all_R = np.concatenate(all_R, axis=0)
        all_w = np.concatenate(all_w, axis=0)
        all_z = np.concatenate(all_z, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Split by domain
        domain0_idx = np.where(all_labels == 0)[0]
        domain1_idx = np.where(all_labels == 1)[0]

        # Create dictionaries for each domain
        domain0_data = {
            'X': all_X[domain0_idx],  # Original data in vector form
            'A': all_A[domain0_idx],  # Harmonized data in vector form
            'R': all_R[domain0_idx],  # Reconstructed data in vector form
            'w': all_w[domain0_idx],  # Embedding of observed data
            'z': all_z[domain0_idx]   # Domain invariant embedding
        }

        domain1_data = {
            'X': all_X[domain1_idx],  # Original data in vector form
            'A': all_A[domain1_idx],  # Harmonized data in vector form
            'R': all_R[domain1_idx],  # Reconstructed data in vector form
            'w': all_w[domain1_idx],  # Embedding of observed data
            'z': all_z[domain1_idx]   # Domain invariant embedding
        }

        # Combined data
        all_data = {
            'X': all_X,               # Original data in vector form
            'A': all_A,               # Harmonized data in vector form
            'R': all_R,               # Reconstructed data in vector form
            'w': all_w,               # Embedding of observed data
            'z': all_z,               # Domain invariant embedding
            'domain_labels': all_labels  # Domain labels
        }

        # Save results
        if save and save_path:
            os.makedirs(save_path, exist_ok=True)
            savemat(os.path.join(save_path, f'{set_name}_domain0.mat'), domain0_data)
            savemat(os.path.join(save_path, f'{set_name}_domain1.mat'), domain1_data)
            savemat(os.path.join(save_path, f'{set_name}_all.mat'), all_data)
            print(f"Saved {set_name} results to {save_path}")

        # Calculate and print some statistics
        print(f"\n{set_name} Statistics:")
        print(f"Domain 0 subjects: {len(domain0_idx)}")
        print(f"Domain 1 subjects: {len(domain1_idx)}")
        print(f"Total subjects: {len(all_labels)}")

        # Calculate mean values for each domain (using means of vectors)
        print("\nMean values:")
        print(f"Domain 0 original: {np.mean(all_X[domain0_idx]):.4f}")
        print(f"Domain 1 original: {np.mean(all_X[domain1_idx]):.4f}")
        print(f"Domain 0 harmonized: {np.mean(all_A[domain0_idx]):.4f}")
        print(f"Domain 1 harmonized: {np.mean(all_A[domain1_idx]):.4f}")
        print(f"Domain 0 reconstructed: {np.mean(all_R[domain0_idx]):.4f}")
        print(f"Domain 1 reconstructed: {np.mean(all_R[domain1_idx]):.4f}")

        # Calculate MSE between original and reconstructed (using vectors)
        domain0_mse = np.mean((all_X[domain0_idx] - all_R[domain0_idx])**2)
        domain1_mse = np.mean((all_X[domain1_idx] - all_R[domain1_idx])**2)
        all_mse = np.mean((all_X - all_R)**2)

        print("\nMSE between original and reconstructed:")
        print(f"Domain 0: {domain0_mse:.4f}")
        print(f"Domain 1: {domain1_mse:.4f}")
        print(f"All subjects: {all_mse:.4f}")
        
        return {
            'domain0': domain0_data,
            'domain1': domain1_data,
            'all': all_data,
            'stats': {
                'domain0_mse': domain0_mse,
                'domain1_mse': domain1_mse,
                'all_mse': all_mse,
                'domain0_count': len(domain0_idx),
                'domain1_count': len(domain1_idx)
            }
        } 