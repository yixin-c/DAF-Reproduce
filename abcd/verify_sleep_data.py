#!/usr/bin/env python
import numpy as np
import scipy.io as sio
import argparse

def verify_mat_file(mat_file):
    print(f"\nVerifying MAT file: {mat_file}")
    mat_data = sio.loadmat(mat_file)

    # Print variables in the updated MAT file
    print('Variables in the updated MAT file:')
    for key in sorted([k for k in mat_data.keys() if not k.startswith('__')]):
        if hasattr(mat_data[key], 'shape'):
            print(f'  {key}: shape {mat_data[key].shape}')
        else:
            print(f'  {key}: not an array')

    # Print the first few rows of sleep data
    print('\nSample of subjects with sleep data:')
    count = 0
    for i in range(len(mat_data['sub_withnet_id'])):
        if not np.isnan(mat_data['sleep_avg_min'][i][0]):
            subject_id = mat_data['sub_withnet_id'][i][0][0]
            sleep_min = mat_data['sleep_avg_min'][i][0]
            sleep_domain = mat_data['sleep_domain'][i][0]
            sleep_numeric = mat_data['sleep_domain_numeric'][i][0]
            print(f"  {subject_id}: {sleep_min} minutes, domain: {sleep_domain}, numeric: {sleep_numeric}")
            count += 1
            if count >= 5:
                break

    # Print statistics
    has_sleep_data = ~np.isnan(mat_data['sleep_avg_min'][:, 0])
    total_with_sleep = np.sum(has_sleep_data)
    total_subjects = len(mat_data['sleep_avg_min'])
    print(f'\nTotal subjects with sleep data: {total_with_sleep} out of {total_subjects} ({total_with_sleep/total_subjects*100:.2f}%)')

    # Count of short vs long sleep domains
    if 'sleep_domain_numeric' in mat_data:
        valid_domains = ~np.isnan(mat_data['sleep_domain_numeric'][:, 0])
        long_sleep = np.sum(mat_data['sleep_domain_numeric'][valid_domains, 0] == 1)
        short_sleep = np.sum(mat_data['sleep_domain_numeric'][valid_domains, 0] == 0)
        print(f'Sleep domains: {short_sleep} short, {long_sleep} long')

def main():
    parser = argparse.ArgumentParser(description='Verify MAT file with sleep data')
    parser.add_argument('--file', type=str, default='/home/yixin/DAF-Reproduce/abcd/new_abcd_cog_with_sleep.mat',
                        help='Path to the MAT file to verify')
    args = parser.parse_args()
    
    verify_mat_file(args.file)

if __name__ == "__main__":
    main() 