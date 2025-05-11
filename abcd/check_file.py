#!/usr/bin/env python
import scipy.io as sio
import numpy as np

# Load the MAT file
mat_file = '/home/yixin/DAF-Reproduce/abcd/new_abcd_cog_with_sleep_filtered.mat'
mat_data = sio.loadmat(mat_file)

# Print variables in the MAT file
print('Variables in the MAT file:')
for key in sorted([k for k in mat_data.keys() if not k.startswith('__')]):
    if hasattr(mat_data[key], 'shape'):
        print(f'  {key}: shape {mat_data[key].shape}')
    else:
        print(f'  {key}: not an array')

# Print sleep domain distribution
valid_domains = ~np.isnan(mat_data['sleep_domain_numeric'][:, 0])
long_sleep = np.sum(mat_data['sleep_domain_numeric'][valid_domains, 0] == 1)
short_sleep = np.sum(mat_data['sleep_domain_numeric'][valid_domains, 0] == 0)
print(f'Sleep domains distribution: {long_sleep} long, {short_sleep} short') 