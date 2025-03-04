import os
import json
import csv

import calculate_semantic_distance
import calculate_perplexity_entropy

# Define the complete list of parameters.
PARAMETERS = ["temperature", "sampling", "context_span", "target_length"]


def get_csv_writer(which_file, data_dir, file_writers):
    """
    Returns a CSV writer for the given file type ("A" or "B").
    The writer is created (with a header) if not already open.
    """
    if which_file not in file_writers:
        filename = f"{which_file}.csv"
        path = os.path.join(data_dir, filename)
        file_existed = os.path.exists(path)
        f = open(path, "a", newline="")
        writer = csv.writer(f)
        if not file_existed:
            if which_file == "A":
                header = PARAMETERS + [
                    "prompt_number",
                    "avg_consec",
                    "avg_all_pairs",
                    "avg_sentence_distances",
                    "wmd_sentence_distances",
                    "avg_entropy_per_section",
                    "avg_perplexity_per_section",
                ]
            else:  # which_file == "B"
                header = PARAMETERS + [
                    "prompt_number",
                    "sentence_number",
                    "avg_prompt_cosine",
                    "avg_prompt_wmd",
                    "all_prompt_sentence_cosine",
                    "all_prompt_sentence_wmd",
                    "all_prompt_sentence_perplexity",
                ]
            writer.writerow(header)
        file_writers[which_file] = (f, writer)
    return file_writers[which_file]


def process_prompt(prompt_idx, params,
                   avg_consec_list, avg_all_pairs_list, avg_sentence_dist_list, wmd_sentence_dist_list,
                   avg_prompt_cosine_list, avg_prompt_wmd_list, all_sent_cosines_list, all_sent_wmds_list,
                   avg_perplexity_list, avg_entropy_list, sentence_perplexity_lists,
                   data_dir, file_writers):
    """
    Process a single prompt for both CSV outputs.
    For CSV A, writes a row if all prompt-level metrics are non-zero.
    For CSV B, iterates over sentence-level metrics and writes a row for each valid sentence.
    """
    # Prompt-level metrics for CSV A.
    a_consec = avg_consec_list[prompt_idx]
    a_allp = avg_all_pairs_list[prompt_idx]
    a_sdist = avg_sentence_dist_list[prompt_idx]
    a_wmd = wmd_sentence_dist_list[prompt_idx]
    a_ent = avg_entropy_list[prompt_idx]
    a_ppl = avg_perplexity_list[prompt_idx]

    if not any(x == 0 for x in [a_consec, a_allp, a_sdist, a_wmd, a_ent, a_ppl]):
        fileA, writerA = get_csv_writer("A", data_dir, file_writers)
        writerA.writerow([
            params["temperature"],
            params["sampling"],
            params["context_span"],
            params["target_length"],
            prompt_idx,
            a_consec,
            a_allp,
            a_sdist,
            a_wmd,
            a_ent,
            a_ppl,
        ])

    # Sentence-level metrics for CSV B.
    b_avg_cos = avg_prompt_cosine_list[prompt_idx]
    b_avg_wmd = avg_prompt_wmd_list[prompt_idx]
    if b_avg_cos == 0 or b_avg_wmd == 0:
        return

    per_sentence_cosines = all_sent_cosines_list[prompt_idx]
    per_sentence_wmds = all_sent_wmds_list[prompt_idx]
    per_sentence_perplexity = sentence_perplexity_lists[prompt_idx]

    for s_idx, (s_cos, s_wmd, s_per) in enumerate(zip(per_sentence_cosines, per_sentence_wmds, per_sentence_perplexity)):
        if s_cos == 0 or s_wmd == 0 or s_per == 0:
            continue
        fileB, writerB = get_csv_writer("B", data_dir, file_writers)
        writerB.writerow([
            params["temperature"],
            params["sampling"],
            params["context_span"],
            params["target_length"],
            prompt_idx,
            s_idx,
            b_avg_cos,
            b_avg_wmd,
            s_cos,
            s_wmd,
            s_per,
        ])


def process_prompts(semantic_results, perplexity_results, params, data_dir, file_writers):
    """
    Processes all prompts (assumed to be 4 per timepoint) using the computed semantic
    and perplexity results, along with the parameter dictionary.
    """
    (
        avg_consec_list,
        avg_all_pairs_list,
        avg_sentence_dist_list,
        wmd_sentence_dist_list,
        avg_prompt_cosine_list,
        avg_prompt_wmd_list,
        all_sent_cosines_list,
        all_sent_wmds_list,
    ) = semantic_results

    (
        avg_perplexity_list,
        avg_entropy_list,
        sentence_perplexity_lists,
        sentence_entropy_lists,  # not used in further processing
    ) = perplexity_results

    for prompt_idx in range(4):
        if prompt_idx >= len(all_sent_cosines_list):
            print(f"Skipping prompt {prompt_idx}: Not enough entries in all_sent_cosines_list")
            continue
        process_prompt(
            prompt_idx,
            params,
            avg_consec_list,
            avg_all_pairs_list,
            avg_sentence_dist_list,
            wmd_sentence_dist_list,
            avg_prompt_cosine_list,
            avg_prompt_wmd_list,
            all_sent_cosines_list,
            all_sent_wmds_list,
            avg_perplexity_list,
            avg_entropy_list,
            sentence_perplexity_lists,
            data_dir,
            file_writers
        )


def process_timepoint(timepoint, t_idx, subject, subgroup, subject_id, varied_param, constants, out_dir, data_dir, file_writers):
    """
    For a given timepoint, builds the complete parameter dictionary using the varied parameter value
    (from the timepoint) and constant values from metadata.
    Then it loads the embeddings and logits files, computes the metrics, and processes each prompt.
    """
    param_value = timepoint.get("varied_param_value", None)
    if param_value is None:
        return

    # Skip timepoints that do not meet the thresholds for temperature or sampling.
    if (varied_param == "temperature" and param_value > 2) or \
       (varied_param == "sampling" and param_value > 0.9):
        return

    # Build a parameters dictionary for all four parameters.
    params = {}
    for param in PARAMETERS:
        if param == varied_param:
            params[param] = param_value
        else:
            params[param] = constants.get(param, None)

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
        return

    print(f"Processing subject {subject_id}, group {subgroup}, timepoint {t_idx}")

    semantic_results = calculate_semantic_distance.run(embeddings_file)
    perplexity_results = calculate_perplexity_entropy.run(logits_file)

    process_prompts(semantic_results, perplexity_results, params, data_dir, file_writers)


def process_metadata(metadata, subgroup, subject, out_dir, data_dir, file_writers):
    """
    Processes the metadata for a given subject subgroup.
    Determines which parameter is varied and uses constant values for the others.
    Then processes each timepoint accordingly.
    """
    varied_param = metadata.get("varied_param", None)
    if varied_param not in PARAMETERS:
        return

    constants = metadata.get("constants", {})
    if (constants.get("temperature", 0) > 2 and varied_param != "temperature") or \
       (constants.get("sampling", 0) > 0.9 and varied_param != "sampling"):
        return

    timepoints = metadata.get("timepoints", [])
    subject_id = metadata.get("subject", "")
    n_timepoints = len(timepoints)

    for t_idx in range(n_timepoints):
        process_timepoint(timepoints[t_idx], t_idx, subject, subgroup, subject_id,
                          varied_param, constants, out_dir, data_dir, file_writers)


def process_subject(subject, subject_path, subgroups_of_interest, out_dir, data_dir, file_writers):
    """
    Iterates over each subgroup in the subject folder and processes the metadata file.
    """
    for subgroup in os.listdir(subject_path):
        if subgroup not in subgroups_of_interest:
            continue

        subgroup_path = os.path.join(subject_path, subgroup)
        if not os.path.isdir(subgroup_path):
            continue

        metadata_file = os.path.join(subgroup_path, "metadata.json")
        if not os.path.exists(metadata_file):
            print(f"Missing metadata.json in {subgroup_path}")
            continue

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        process_metadata(metadata, subgroup, subject, out_dir, data_dir, file_writers)


def close_all_writers(file_writers):
    """
    Closes all open CSV file pointers.
    """
    for f_obj, _ in file_writers.values():
        f_obj.close()


def run_combined_analysis():
    base_dir = '/home/ubuntu/emilia/pel_output_unif'
    data_dir = '/home/ubuntu/emilia/csv_data/data_unif_tempconstr_all-param'
    out_dir = os.path.join(base_dir, 'Outputs')
    metadata_dir = os.path.join(base_dir, 'Metadata')

    # Use a single CSV writer for each group: "A" and "B".
    file_writers = {}

    subgroups_of_interest = ["a", "b", "c", "d"]

    for subject in os.listdir(metadata_dir):
        subject_path = os.path.join(metadata_dir, subject)
        if not os.path.isdir(subject_path) or not subject.startswith("s"):
            continue
        process_subject(subject, subject_path, subgroups_of_interest, out_dir, data_dir, file_writers)

    close_all_writers(file_writers)


if __name__ == "__main__":
    run_combined_analysis()