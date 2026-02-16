#!/usr/bin/env python3
"""
Script to split xm_extra_vars.csv into separate files by windmill element.
"""
import pandas as pd
import os

def split_csv_by_element(input_file, output_dir):
    """
    Split CSV file by Element column, creating one file per windmill.
    
    Args:
        input_file: Path to the input CSV file
        output_dir: Directory where the split files will be saved
    """
    print(f"Reading {input_file}...")
    
    # Read the CSV file in chunks to handle large files efficiently
    chunk_size = 100000
    first_chunk = True
    
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        # Get unique elements in this chunk
        elements = chunk['Element'].unique()
        
        for element in elements:
            element_data = chunk[chunk['Element'] == element]
            output_file = os.path.join(output_dir, f'{element}.csv')
            
            # Write header only for the first chunk
            if first_chunk:
                element_data.to_csv(output_file, mode='w', index=False)
            else:
                # Check if file exists, if so append without header
                if os.path.exists(output_file):
                    element_data.to_csv(output_file, mode='a', header=False, index=False)
                else:
                    element_data.to_csv(output_file, mode='w', index=False)
        
        first_chunk = False
        print(f"Processed chunk...")
    
    print(f"\nSplit complete! Files saved to: {output_dir}")
    
    # List the created files
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and 'ag' in f]
    print(f"\nCreated {len(csv_files)} files:")
    for f in sorted(csv_files):
        file_path = os.path.join(output_dir, f)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"  - {f} ({file_size:.2f} MB)")

if __name__ == "__main__":
    input_file = "data/xm_extra_vars.csv"
    output_dir = "data/split_by_element"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    split_csv_by_element(input_file, output_dir)
