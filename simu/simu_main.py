import json
import os
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

    # Iterate through each cohort
    for cohort_name, cohort_config in config.cohorts.items():
        print(f"Processing cohort: {cohort_name}")
        cohort_dir = os.path.join(config.directory, cohort_name)
        os.makedirs(cohort_dir, exist_ok=True)
        print(cohort_dir)
        progress_file = os.path.join(cohort_dir, "progress.json")
        progress = setup.load_progress(progress_file)

        # Generate data for each subject in the cohort
        for subject in range(config.subjects_per_cohort):
            if progress.get(str(subject), {}).get("completed", False):
                print(f"Skipping subject {subject} in {cohort_name}, already completed.")
                continue

            print(f"Generating data for subject {subject} in {cohort_name}")
            progress[str(subject)] = progress.get(str(subject), {"timepoints": {}, "completed": False})
            subject_file = os.path.join(cohort_dir, f"subject_{subject}.json")

            # Process each timepoint for the subject
            for timepoint in range(config.timepoints_per_subject):
                if progress[str(subject)]["timepoints"].get(str(timepoint), False):
                    print(f"Skipping timepoint {timepoint} for subject {subject}, already completed.")
                    continue

                # Unpack subject-specific details
                constants, varied_param, varied_param_mean, varied_param_variance = generate_parameter.ParameterGenerator.subject_sample(config, cohort_name).values()

                # Generate timepoint-specific value for the varied parameter
                varied_param_value = generate_parameter.ParameterGenerator.timepoint_sample(
                    varied_param_mean, varied_param_variance
                )

                # Combine constant and timepoint-specific parameters
                parameters = {**constants, varied_param: varied_param_value}
                parameters = generate_parameter.ParameterGenerator.clean_parameters(parameters)

                # Calculate well-being factors, including state score
                wellbeing_factors = generate_parameter.ParameterGenerator.state_sample(
                    parameters, config.parameter_rules
                )

                # Create a record for the timepoint
                record = {
                    "cohort": cohort_name,
                    "subject": subject,
                    "timepoint": timepoint,
                    "varied_param": varied_param,
                    "varied_param_mean": varied_param_mean,
                    "varied_param_variance": varied_param_variance,
                    "parameters": parameters,
                    "wellbeing_factors": wellbeing_factors,
                }

                # Generate generation arguments for TextGenerator
                generation_arguments = generate_parameter.ParameterGenerator.generate_parameters(parameters, setup)
                text_generator = generate_text.TextGenerator(setup, config.prompts, parameters, generation_arguments)
                generated_text = text_generator.out

                with open(subject_file, "a") as file:
                    json.dump(record, file, indent=4)
                    file.write("\n")
                    
                    json.dump({"generated_text": generated_text}, file, indent=4)
                    file.write("\n")

                # Update progress for this timepoint
                progress[str(subject)]["timepoints"][str(timepoint)] = True
                setup.save_progress(progress, progress_file)

            # Mark subject as completed
            progress[str(subject)]["completed"] = True
            setup.save_progress(progress, progress_file)

    # Calculate and print the total execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")