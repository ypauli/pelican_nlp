import os
import json
import csv

import calculate_semantic_distance
import calculate_perplexity_entropy

def run_combined_analysis():
    base_dir = '/home/ubuntu/emilia/pel_output_unif_test'
    data_dir = '/home/ubuntu/emilia/PELICAN/pelican/simulation/simu_analysis/data'
    out_dir = os.path.join(base_dir, 'Outputs')
    metadata_dir = os.path.join(base_dir, 'Metadata')
    
    # A place to store open file pointers & CSV writers:
    file_writers = {}
    
    # We'll only handle these "varied_param" values
    valid_varied_params = ["temperature", "sampling", "context_span", "target_length"]
    subgroups_of_interest = ["a", "b", "c", "d"]
    
    # A helper function to ensure we have an open CSV file + writer
    def get_csv_writer(varied_param, which_file):
        """
        which_file is either 'A' or 'B', meaning "group-a.csv" vs "group-b.csv".
        This returns a csv.writer for that file, creating it if necessary.
        """
        if varied_param not in file_writers:
            file_writers[varied_param] = {}
        
        if which_file not in file_writers[varied_param]:
            # Build the filepath, e.g. data_dir/temperature-a.csv or data_dir/temperature-b.csv
            filename = f"{varied_param}-{which_file}.csv"
            path = os.path.join(data_dir, filename)
            
            # If which_file == 'A', we append or create, but start with a header if file is empty
            # We'll open in "a" mode to accumulate data, but check if file is new:
            file_existed = os.path.exists(path)
            f = open(path, "a", newline="")
            wr = csv.writer(f)
            
            # Write header if file didn't exist
            if not file_existed:
                if which_file == "A":
                    # Columns for "group-a.csv"
                    header = [
                        "varied_param_value",
                        "prompt_number",
                        "avg_consec",
                        "avg_all_pairs",
                        "avg_sentence_distances",
                        "wmd_sentence_distances",
                        "avg_entropy_per_section",
                        "avg_perplexity_per_section"
                    ]
                else:
                    # Columns for "group-b.csv"
                    header = [
                        "varied_param_value",
                        "prompt_number",
                        "sentence_number",
                        "avg_prompt_cosine",
                        "avg_prompt_wmd",
                        "all_prompt_sentence_cosine",
                        "all_prompt_sentence_wmd", 
                    ]
                wr.writerow(header)
            
            file_writers[varied_param][which_file] = (f, wr)
        
        return file_writers[varied_param][which_file]
    
    # For each subject folder, read metadata from each subgroup "a"/"b"/"c"/"d"
    for subject in os.listdir(metadata_dir):
        subject_path = os.path.join(metadata_dir, subject)
        if not os.path.isdir(subject_path):
            continue
        
        if not subject.startswith("s"):
            continue
        
        # Now check subgroups a,b,c,d under this subject
        for subgroup in os.listdir(subject_path):
            if subgroup not in subgroups_of_interest:
                continue
            
            subgroup_path = os.path.join(subject_path, subgroup)
            if not os.path.isdir(subgroup_path):
                continue
            
            # Read the metadata.json
            metadata_file = os.path.join(subgroup_path, "metadata.json")
            if not os.path.exists(metadata_file):
                print(f"Missing metadata.json in {subgroup_path}")
                continue
            
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            
            #   Metadata fields: 
            #   metadata["varied_param"] might be "temperature"
            #   metadata["timepoints"] is a list
            #   metadata["subject"] is the subject id
            #   metadata["group"] is the single char "a" or "b" etc.
            
            varied_param = metadata.get("varied_param", None)
            if varied_param not in valid_varied_params:
                continue
            
            timepoints = metadata.get("timepoints", [])
            subject_id = metadata.get("subject", "")
            # group_in_metadata = metadata.get("group", "")  # "a" or "b" etc.
            n_timepoints = len(timepoints)
            
            # For each timepoint, find the corresponding embeddings and logits
            for t_idx in range(n_timepoints):
                # Build the file paths
                embeddings_file = os.path.join(
                    out_dir, subject, "ses-0", subgroup,
                    f"sub-{subject_id}_ses-0_group-{subgroup}_timepoint-{t_idx}_results_embeddings.csv"
                )
                logits_file = os.path.join(
                    out_dir, subject, "ses-0", subgroup,
                    f"sub-{subject_id}_ses-0_group-{subgroup}_timepoint-{t_idx}_results_logits.csv"
                )
                
                if not (os.path.exists(embeddings_file) and os.path.exists(logits_file)):
                    print("Missing embeddings or logits file for", subject_id, subgroup, t_idx)
                    continue
                
                print(f"Processing subject {subject_id}, group {subgroup}, timepoint {t_idx}")
                
                # Calculate Semantic Distances, Perplexity and Entropy
                (
                    avg_consec_list,
                    avg_all_pairs_list,
                    avg_sentence_dist_list,
                    wmd_sentence_dist_list,
                    avg_prompt_cosine_list,
                    avg_prompt_wmd_list,
                    all_sent_cosines_list,
                    all_sent_wmds_list
                ) = calculate_semantic_distance.run(embeddings_file)
                
                (
                    avg_perplexity_list,
                    avg_entropy_list,
                    sentence_perplexity_lists,
                    sentence_entropy_lists,
                ) = calculate_perplexity_entropy.run(logits_file)
                
                # This script now loops over the 4 prompts for each timepoint.
                param_value = timepoints[t_idx].get("varied_param_value", None)
                
                # If param_value is None, we skip (or default it).
                if param_value is None:
                    print("No varied_param value found at ", t_idx, "for", subject_id, subgroup)
                    continue
                
                for prompt_idx in range(3):
                    # 2) Build row for FILE A
                    #    We skip writing if ANY relevant field is zero (meaning no text after prompt).
                    a_consec  = avg_consec_list[prompt_idx]
                    a_allp    = avg_all_pairs_list[prompt_idx]
                    a_sdist   = avg_sentence_dist_list[prompt_idx]
                    a_wmd     = wmd_sentence_dist_list[prompt_idx]
                    a_ent     = avg_entropy_list[prompt_idx]
                    a_ppl     = avg_perplexity_list[prompt_idx]
                    
                    # In case any of the values are zero, do not include the data. So let's define "relevant" for file A:
                    relevant_for_A = [a_consec, a_allp, a_sdist, a_wmd, a_ent, a_ppl]
                    if any(x == 0 for x in relevant_for_A):
                        # Skip
                        pass
                    else:
                        # Write a row to the "A" CSV for (varied_param)
                        (fileA, writerA) = get_csv_writer(varied_param, "A")
                        
                        writerA.writerow([
                            param_value,     # varied_param_value
                            prompt_idx,      # prompt_number
                            a_consec,        # Avg. consecutive token distances
                            a_allp,          # Avg. all-pair token distances
                            a_sdist,         # Avg. sentence-to-sentence distances (cosine)
                            a_wmd,           # Avg. sentence-to-sentence WMD
                            a_ent,           # Avg. Entropy per Section
                            a_ppl            # Avg. Perplexity per Section
                        ])
                        
                    
                    # 3) Build rows for FILE B, one row per sentence in that prompt
                    #    We skip if any "prompt-level" value is 0 as well.
                    b_avg_cos = avg_prompt_cosine_list[prompt_idx]
                    b_avg_wmd = avg_prompt_wmd_list[prompt_idx]
                    
                    # all_sent_cosines_list[prompt_idx] => e.g. [cos_s1, cos_s2, ...]
                    # all_sent_wmds_list[prompt_idx]     => e.g. [wmd_s1, wmd_s2, ...]
                    # We'll loop over each sentence_number
                    
                    # If the "avg_prompt_cosine" or "avg_prompt_wmd" is zero, skip them
                    if b_avg_cos == 0 or b_avg_wmd == 0:
                        continue

                    per_sentence_cosines = all_sent_cosines_list[prompt_idx]
                    per_sentence_wmds    = all_sent_wmds_list[prompt_idx]
                    # per_sentence_perplexity = sentence_perplexity_lists[prompt_idx]
                    # per_sentence_entropy = sentence_entropy_lists[prompt_idx]
                    
                    # They should have the same length
                    for s_idx, (s_cos, s_wmd) in enumerate(zip(per_sentence_cosines, per_sentence_wmds)):
                        # If s_cos == 0 or s_wmd == 0, skip just that sentence row
                        if s_cos == 0 or s_wmd == 0:
                            continue
                        
                        # Write to "B" CSV
                        (fileB, writerB) = get_csv_writer(varied_param, "B")
                        writerB.writerow([
                            param_value,       # varied_param_value
                            prompt_idx,        # prompt_number
                            s_idx,             # sentence_number
                            b_avg_cos,         # Avg. prompt-to-sentence distances (cosine)
                            b_avg_wmd,         # Avg. prompt-to-sentence WMD
                            s_cos,             # Avg. cosine distance from prompt for each sentence
                            s_wmd,             # Avg. WMD distance from prompt for each sentence
                        ])
                            
    # After processing everything, close all the open files
    for vp_dict in file_writers.values():
        for which_file_tuple in vp_dict.values():
            f_obj, _ = which_file_tuple
            f_obj.close()

if __name__ == "__main__":
    run_combined_analysis()