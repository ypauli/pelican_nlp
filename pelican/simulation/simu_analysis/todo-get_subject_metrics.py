import os
import csv
import json

base_dir = '/home/ubuntu/emilia/PELICAN/pelican/simulation/simu_output'
out_dir = os.path.join(base_dir, 'Outputs')
metadata_dir = os.path.join(base_dir, 'Metadata')

csv_dir = '/home/ubuntu/emilia/PELICAN/simu_analysis/data'
csv_rows = {"timepoint", "temperature", "sampling", "context", "length", "semantic_distance", "perplexity", "entropy"}

for subject in os.listdir(metadata_dir):
    for group in os.listdir(os.path.join(metadata_dir, subject)):
        
        # Initialize CSV
        data_csv = os.path.appen(csv_dir, f"data_sub-{subject}_{group}.csv")
        if os.path.exists(data_csv):
            continue
        with open(data_csv, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=csv_rows.keys())
            writer.writeheader()
        
        # Get subject metadata
        metadata = os.path.join(metadata_dir, subject, group, "metadata.json")
        with open(metadata, 'r') as f:
            metadata = json.load(f)
            
            row = None
            
            # Calculate metrics/timepoint
            timepoints = metadata["timepoints"]
            for tp in timepoints:
                row["metadata["varied_param"]"] = tp["varied_parameter"]

                
                # Write column names
                writer.writerow(row)  # Write data
