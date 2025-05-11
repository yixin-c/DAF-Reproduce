#!/usr/bin/env python
import numpy as np
import pandas as pd
import scipy.io as sio
import os
import sys
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Add sleep data to ABCD MAT file')
    parser.add_argument('--remove-missing', action='store_true', 
                        help='Remove subjects without sleep data')
    parser.add_argument('--output', type=str, 
                        default='/home/yixin/DAF-Reproduce/abcd/new_abcd_cog_with_sleep.mat',
                        help='Output MAT file path')
    args = parser.parse_args()

    # Path to the files
    mat_file = '/home/yixin/DAF-Reproduce/abcd/new_abcd_cog.mat'
    sleep_file = '/home/yixin/DAF-Reproduce/abcd/abcd_sleep_time_per_child.csv'

    # Output file
    output_file = args.output

    print(f"Loading MAT file: {mat_file}")
    try:
        # Load the MAT file
        mat_data = sio.loadmat(mat_file)
        
        # Print variables in the MAT file
        print("Variables in the MAT file:")
        for key in mat_data.keys():
            if not key.startswith('__'):  # Skip metadata variables
                print(f"  {key}: {type(mat_data[key])} with shape {mat_data[key].shape if hasattr(mat_data[key], 'shape') else 'N/A'}")
        
        # Load the sleep data
        print(f"\nLoading sleep data from: {sleep_file}")
        sleep_data = pd.read_csv(sleep_file)
        print(f"Sleep data shape: {sleep_data.shape}")
        
        # Extract subject IDs from the MAT file
        subject_ids = mat_data['sub_withnet_id']
        num_subjects = len(subject_ids)
        print(f"\nNumber of subjects in MAT file: {num_subjects}")
        
        # Prepare arrays for the new covariates
        sleep_avg_min = np.full((num_subjects, 1), np.nan)
        sleep_domain = np.zeros((num_subjects, 1), dtype=object)
        sleep_domain_numeric = np.full((num_subjects, 1), np.nan)  # 0 for short, 1 for long
        
        # Track which subjects have sleep data for potential removal
        has_sleep_data = np.zeros(num_subjects, dtype=bool)
        
        # Count for matching statistics
        match_count = 0
        
        # Match subject IDs and add sleep data
        for i in range(num_subjects):
            # Extract subject ID (removing array wrapper and converting to string)
            subject_id = subject_ids[i][0][0]
            
            # Find this subject in the sleep data
            sleep_rows = sleep_data[sleep_data['subjectkey'] == subject_id]
            
            if not sleep_rows.empty:
                match_count += 1
                has_sleep_data[i] = True
                
                # Use the first row if multiple entries exist
                sleep_row = sleep_rows.iloc[0]
                
                # Add sleep minutes
                sleep_avg_min[i, 0] = sleep_row['sleep_avg_min']
                
                # Add sleep domain as string
                sleep_domain[i, 0] = sleep_row['sleep_domain']
                
                # Add sleep domain as numeric (0 for short, 1 for long)
                sleep_domain_numeric[i, 0] = 1 if sleep_row['sleep_domain'] == 'long' else 0
        
        # Add the new data to the MAT file
        mat_data['sleep_avg_min'] = sleep_avg_min
        mat_data['sleep_domain'] = sleep_domain
        mat_data['sleep_domain_numeric'] = sleep_domain_numeric
        
        print(f"\nMatched {match_count} out of {num_subjects} subjects ({match_count/num_subjects*100:.2f}%)")
        
        # Remove subjects without sleep data if requested
        if args.remove_missing:
            print(f"\nRemoving {num_subjects - match_count} subjects without sleep data...")
            
            # Create new data with only subjects who have sleep data
            new_mat_data = {}
            for key in mat_data.keys():
                if key.startswith('__'):  # Copy metadata as is
                    new_mat_data[key] = mat_data[key]
                elif key in ['sub_withnet_id', 'sleep_avg_min', 'sleep_domain', 'sleep_domain_numeric']:
                    # These are column vectors, so we filter rows
                    new_mat_data[key] = mat_data[key][has_sleep_data]
                elif key == 'count_network':
                    # This is a 3D array with subjects in the 3rd dimension
                    new_mat_data[key] = mat_data[key][:, :, has_sleep_data]
                elif key in ['demographx', 'nih_cogn']:
                    # These are 2D arrays with subjects in rows
                    new_mat_data[key] = mat_data[key][has_sleep_data, :]
                else:
                    # For other variables, copy as is
                    new_mat_data[key] = mat_data[key]
            
            # Use the filtered data
            mat_data = new_mat_data
            print(f"New data dimensions after removal:")
            for key in mat_data.keys():
                if not key.startswith('__'):  # Skip metadata variables
                    print(f"  {key}: shape {mat_data[key].shape if hasattr(mat_data[key], 'shape') else 'N/A'}")
        
        # Save the updated MAT file
        print(f"\nSaving updated MAT file to: {output_file}")
        sio.savemat(output_file, mat_data)
        print("Done!")
        
    except Exception as e:
        print(f"Error processing files: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 