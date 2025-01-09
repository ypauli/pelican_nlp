import json
import os
import numpy as np

from config import Config
from simu import ParameterGenerator
from simu import PipelineSetup
from simu import TextGenerator

if __name__ == "__main__":
    # Initialize configuration and pipeline setup
    config = Config()
    setup = PipelineSetup(config)

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
                    print(f"Timepoints per subject: {config.timepoints_per_subject}")  # Debugging
                    
                    
                    # Generate timepoint-specific value for the varied parameter
                    varied_param_value = ParameterGenerator.timepoint_sample(
                        varied_param_mean, varied_param_variance
                    )

                    # Combine constant and timepoint-specific parameters
                    parameters = {**constants, varied_param: varied_param_value}
                    parameters["context_span"] = round(parameters["context_span"])
                    parameters["target_length"] = round(parameters["target_length"])

                    # Calculate well-being factors
                    wellbeing_factors = ParameterGenerator.wellbeing_sample(
                        parameters, config.parameter_rules
                    )

                    # Create a record for the timepoint
                    record = {
                        "subject": subject,
                        "timepoint": timepoint,
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