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
logging.basicConfig(filename='./logs/daf.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Needed for reproducibility in CUDA ≥10.2

set_seed()

# Unsupervised DAF

## Load brain connectome data (adjacency matrices)

dat_abcd = loadmat('../abcd/new_abcd_cog.mat')
dat_hcp = loadmat('../hcp/HCP_subcortical_CMData_desikan.mat')
tensor_abcd = dat_abcd['count_network']
tensor_hcp = dat_hcp['loaded_tensor_sub']

# whether to apply log transformation to the adjacency matrices)
log=True

set_seed(1972014)
# np.random.seed(123456)
# orders = np.random.permutation(np.arange(68))
# offset = 18994
net_abcd = []
for i in range(tensor_abcd.shape[2]):
    ith = np.float32(tensor_abcd[:,:,i])
    if log:
        ith = np.log(ith+1)
    ith = ith.reshape(68*68)
    net_abcd.append(ith)
n_abcd = len(net_abcd)

net_hcp = []
for i in range(tensor_hcp.shape[3]):
    ith = np.float32(tensor_hcp[:,:,0,i] + np.transpose(tensor_hcp[:,:,0,i]))
    ith = ith[18:86, 18:86]
    if log:
        ith = np.log(ith+1)
#     np.fill_diagonal(ith,np.mean(ith, 0))
    ith = ith.reshape(68*68)
    net_hcp.append(ith)
n_hcp = len(net_hcp)

## either sample or take all data from ABCD and HCP
if sample:
    _, id_abcd = train_test_split(range(n_abcd), test_size=892/n_abcd, random_state=42) # big 5079, small 173
    _, id_hcp = train_test_split(range(n_hcp), test_size=173/n_hcp, random_state=42) # big 5079, small 173
else:
    id_abcd = list(range(n_abcd))
    id_hcp = list(range(n_hcp))

print('abcd',len(id_abcd))
print('hcp',len(id_hcp))

net_abcd = [net_abcd[i] for i in id_abcd]
net_hcp = [net_hcp[i] for i in id_hcp]
net_all = net_abcd + net_hcp

## preprocess

if log:
    net_all = torch.stack([torch.Tensor(i) for i in net_all]).view(-1,1,68,68)
else:
    # If no log transformation, scale data to range [-1,1] (default in cycle gan)
    offset = np.max(net_all)/2 # 9497
    net_all = torch.stack([torch.Tensor((i-offset)/offset) for i in net_all]).view(-1,1,68,68)

n_abcd, n_hcp = len(id_abcd), len(id_hcp)
n = n_abcd + n_hcp
# domain label: ABCD(0) HCP(1)
labels = torch.cat((torch.tensor([0]).expand(n_abcd,1), torch.tensor([1]).expand(n_hcp,1)))

dat_abcd['nih_cogn'].shape

## Data loader

set_seed(1972034)
nepoch = 300
dataset = utils.TensorDataset(net_all, labels)
# split train and validataion 
train_id, test_id = train_test_split(list(range(n)),train_size=0.9, random_state=42)
train_dataset = utils.Subset(dataset, train_id)
test_dataset = utils.Subset(dataset, test_id)
train_loader = utils.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = utils.DataLoader(test_dataset, batch_size=len(test_id), shuffle=False)