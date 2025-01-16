import os
import json
import time
import simu_config
import generate_parameter
import setup_pipeline
import generate_text

if __name__ == "__main__":
    # Initialize configuration and pipeline setup
    config = simu_config.SimuConfig()
    setup = setup_pipeline.PipelineSetup(config)
    start_time = time.time()
    
    # Initialize the output directory
    if not os.path.exists(config.directory):
        os.makedirs(config.directory, exist_ok=True)
    subject_dir = os.path.join(config.directory, f"Subjects")
    metadata_dir = os.path.join(config.directory, f"Metadata")

    # Initialize the progress file
    progress_file = os.path.join(config.directory, "progress.json")

    # Check if progress file exists
    if os.path.exists(progress_file):
        with open(progress_file, "r") as file:
            progress = json.load(file)
    else:
        # Initialize an empty progress structure and create the file
        progress = {"subjects": {}}
        with open(progress_file, "w") as file:
            json.dump(progress, file, indent=4)

    # Iterate through each subject
    for subject in range(config.subjects):
                    
        # Initialize subject constants
        constants = generate_parameter.ParameterGenerator.subject_sample(config)
        
        # Initialize subject progress if it doesn't exist, else get constants
        if str(subject) not in progress["subjects"]:
            progress["subjects"][str(subject)] = {"cohorts": {}}
        else: 
            # Fetch constants from the metadata file if it exists
            metadata_file_path = os.path.join(metadata_dir, f"subject_{subject}", "metadata.json")
            if os.path.exists(metadata_file_path):
                with open(metadata_file_path, "r") as metadata_file:
                    subject_metadata = json.load(metadata_file)
                    constants = subject_metadata.get("constants", {})
            else:
                print(f"Metadata file not found for subject {subject}.")
                constants = generate_parameter.ParameterGenerator.subject_sample(config)
        
        # Process each cohort
        for cohort_name, cohort_config in config.cohorts.items():
            print(f"Processing subject: {subject}, cohort: {cohort_name}")

            # Create subject-specific directory for cohort and metadata.json file for this subject and cohort
            cohort_dir = os.path.join(subject_dir, f"subject_{subject}", cohort_name)
            cohort_metadata_dir = os.path.join(metadata_dir, f"subject_{subject}", cohort_name)
            os.makedirs(cohort_dir, exist_ok=True)
            os.makedirs(cohort_metadata_dir, exist_ok=True)

            metadata_json_file = os.path.join(cohort_metadata_dir, "metadata.json")

            # Initialize metadata information
            cohort_data = generate_parameter.ParameterGenerator.cohort_sample(cohort_config, constants)

            # Create or overwrite the metadata file
            metadata = {
                "subject": subject,
                "cohort": cohort_name,
                "constants": constants,
                "varied_param": cohort_data["varied_parameter"],
                "mean": cohort_data["varied_param_mean"],
                "variance": cohort_data["varied_param_variance"],
                "timepoints": []
            }

            with open(metadata_json_file, "w") as meta_file:
                json.dump(metadata, meta_file, indent=4)

            # Process each timepoint for the subject in this cohort
            for timepoint in range(config.timepoints):
                # Check if the timepoint has already been processed
                if progress["subjects"][str(subject)]["cohorts"].get(cohort_name, {}).get(str(timepoint), False):
                    print(f"Skipping timepoint {timepoint} for subject {subject} in cohort {cohort_name}, already completed.")
                    continue

                print(f"Generating data for subject {subject}, cohort {cohort_name}, timepoint {timepoint}")

                # Generate timepoint-specific value for the varied parameter
                varied_param_value = generate_parameter.ParameterGenerator.timepoint_sample(
                    cohort_data["varied_param_mean"], cohort_data["varied_param_variance"]
                )
                parameters = {**constants, cohort_data["varied_parameter"]: varied_param_value}
                parameters = generate_parameter.ParameterGenerator.clean_parameters(parameters)

                # Generate wellbeing factors for the timepoint
                wellbeing_factors = generate_parameter.ParameterGenerator.state_sample(parameters, config.parameter_rules)

                # Create the text for this timepoint
                generation_arguments = generate_parameter.ParameterGenerator.generate_arguments(parameters, setup)
                text_generator = generate_text.TextGenerator(setup, config.prompts, parameters, generation_arguments)
                generated_text = text_generator.out

                # Create a new file for each timepoint
                timepoint_txt_file = os.path.join(cohort_dir, f"timepoint_{timepoint}.txt")

                with open(timepoint_txt_file, "w") as timepoint_file:
                    for idx, prompt in enumerate(generated_text):
                        timepoint_file.write(f"New Prompt: prompt_{idx}:\n")
                        timepoint_file.write(f"{generated_text[prompt]}\n\n")

                # Append the timepoint data to the metadata JSON
                metadata["timepoints"].append({
                    "timepoint": timepoint,
                    "varied_param_value": varied_param_value,
                    "wellbeing_factors": wellbeing_factors
                })

                # Update progress for this timepoint
                if cohort_name not in progress["subjects"][str(subject)]["cohorts"]:
                    progress["subjects"][str(subject)]["cohorts"][cohort_name] = {}

                progress["subjects"][str(subject)]["cohorts"][cohort_name][str(timepoint)] = True

            # Save progress after processing the cohort for the subject
            with open(progress_file, "w") as file:
                json.dump(progress, file, indent=4)

            # Write the updated metadata JSON file with timepoints data
            with open(metadata_json_file, "w") as meta_file:
                json.dump(metadata, meta_file, indent=4)

    # Calculate and print the total execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")