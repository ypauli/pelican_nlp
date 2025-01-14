import json
import os
import numpy as np
import time

from config import Config
from simu import ParameterGenerator
from simu import PipelineSetup
from simu import TextGenerator

if __name__ == "__main__":
    # Initialize configuration and pipeline setup
    config = Config()
    setup = PipelineSetup(config)
    start_time = time.time()

    # Iterate through each cohort
    for cohort_name, cohort_config in config.cohorts.items():
        print(f"Processing cohort: {cohort_name}")
        cohort_dir = os.path.join("simu_output", cohort_name)
        os.makedirs(cohort_dir, exist_ok=True)

        # Generate data for each subject in the cohort
        for subject in range(config.subjects_per_cohort):
            print(f"Generating data for subject {subject} in {cohort_name}")

            # Unpack subject-specific details
            constants, varied_param, varied_param_mean, varied_param_variance = ParameterGenerator.subject_sample(config, cohort_name).values()

            # Subject-specific data storage
            subject_file = os.path.join(cohort_dir, f"subject_{subject}.json")
            
            with open(subject_file, "w") as file:
                for timepoint in range(config.timepoints_per_subject):
                    # Generate timepoint-specific value for the varied parameter
                    varied_param_value = ParameterGenerator.timepoint_sample(
                        varied_param_mean, varied_param_variance
                    )

                    # Combine constant and timepoint-specific parameters
                    parameters = {**constants, varied_param: varied_param_value}
                    
                    # Ensure the values are valid
                    parameters["temperature"] = np.clip(parameters["temperature"], 0.2, 10)
                    parameters["sampling"] = np.clip(parameters["sampling"], 0.2, 0.98)
                    parameters["context_span"] = round(np.clip(parameters["context_span"], 5, 500))
                    parameters["target_length"] = round(np.clip(parameters["target_length"], 5, 1000))

                    # Calculate well-being factors, including state score
                    wellbeing_factors = ParameterGenerator.state_sample(
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
                    json.dump(record, file, indent=4)
                    file.write("\n")
                    
                    # Generate generation arguments for TextGenerator
                    generation_arguments = ParameterGenerator.generate_parameters(parameters, setup)
                    
                    # Generate text using the sampled parameters
                    text_generator = TextGenerator(setup, config.prompts, parameters, generation_arguments)
                    generated_text = text_generator.out
                    
                    json.dump({"generated_text": generated_text}, file, indent=4)
                    file.write("\n")
                    
    # Calculate and print the total execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")