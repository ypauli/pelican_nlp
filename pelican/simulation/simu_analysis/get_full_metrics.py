import os
import re
import csv
import json
import calculate_semantic_distance
import calculate_perplexity
import calculate_entropy

# Define directories
base_dir = '/home/ubuntu/data/simu_output'
data_dir = '/home/ubuntu/data/data_full_prompt'

out_dir = os.path.join(base_dir, 'Outputs')
metadata_dir = os.path.join(base_dir, 'Metadata')
csv_rows = ["temperature", "sampling", "context_span", "target_length","semantic_distance", "perplexity", "entropy"]
n_timepoints = 52
n_prompts = 4

for group in ["temperature", "sampling", "context_span", "target_length"]:
    for prompt in range(n_prompts):
        
        data_file = os.path.join(data_dir, f"{group}-{prompt}.csv")
        with open(data_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            # csv_rows[0] = group
            writer.writerow(csv_rows)
            
for subject in os.listdir(metadata_dir):
    
    if not subject.startswith("s"):
        print(f"Skipping directory {group}")
        continue
    
    for group in os.listdir(os.path.join(metadata_dir, subject)):
        
        if not group.startswith(("a", "b", "c", "d")):
            print(f"Skipping directory {group}")
            continue
        
        # Get subject metadata
        metadata_file = os.path.join(metadata_dir, subject, group, "metadata.json")
        print(metadata_file)
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            timepoints = metadata["timepoints"]
            subject_id = metadata["subject"]
            group = metadata["group"]
            
            # Iterate over timepoints
            for timepoint in range(n_timepoints):
                
                embeddings_file = os.path.join(out_dir, subject, f"ses-0", group, f"sub-{subject_id}_ses-0_group-{group}_timepoint-{timepoint}_results_embeddings.csv")
                logits_file = os.path.join(out_dir, subject, f"ses-0", group, f"sub-{subject_id}_ses-0_group-{group}_timepoint-{timepoint}_results_logits.csv")
                
                if not (os.path.exists(embeddings_file) and os.path.exists(logits_file)):
                    print("Files not found")  
                    print(embeddings_file)
                    print(logits_file)  
                    continue
                    
                print(f"Processing timepoint {timepoint} for subject {subject_id} in group {group}")
                
                semantic_distance = calculate_semantic_distance.run(embeddings_file)
                perplexity = calculate_perplexity.run(logits_file)
                entropy = calculate_entropy.run(logits_file)
                
                 # If one of the returned lists is not of length 4, ignore datapoint
                if len(semantic_distance) != 4 or len(perplexity) != 4 or len(entropy) != 4:
                    continue
                
                # Get metadata and varied parameter values
                row = [metadata["constants"]["temperature"], metadata["constants"]["sampling"], metadata["constants"]["context_span"], metadata["constants"]["target_length"]]
                varied_param = metadata["varied_param"]
                varied_param_value = timepoints[timepoint]["varied_param_value"]
                
                print(row)
                
                # Save calculated metrics to CSV
                for prompt in range(n_prompts):
                    data_file = os.path.join(data_dir, f"{varied_param}-{prompt}.csv")
                    
                    with open(data_file, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        
                        curr_row = row

                        varied_param_index = csv_rows.index(varied_param)
                        curr_row[varied_param_index] = varied_param_value
                        curr_row.append(semantic_distance[prompt])
                        curr_row.append(perplexity[prompt])
                        curr_row.append(entropy[prompt])
                        
                        print(row)
                        writer.writerow(row)
                        
                        