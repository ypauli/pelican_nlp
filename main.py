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
        
        # Ensure covariance matrix is PSD
        cohort_config["covariance_matrix"] = setup.nearest_psd(cohort_config["covariance_matrix"])
        try:
            np.linalg.cholesky(cohort_config["covariance_matrix"])
            print("Matrix is PSD")
        except np.linalg.LinAlgError:
            print("Matrix is still not PSD")

        # Generate data for each subject in the cohort
        for subject in range(config.subjects_per_cohort):
            subject_file = os.path.join(cohort_dir, f"subject_{subject}.json")

            with open(subject_file, "w") as file:
                for timepoint in range(config.timepoints_per_subject):
                    # Sample parameters for the current cohort
                    parameters, generation_arguments = ParameterGenerator.generate_parameters(config, setup, cohort_name)

                    # Derive well-being and related factors from parameters
                    wellbeing_factors = {
                        key: rule(parameters) for key, rule in config.parameter_rules.items()
                    }

                    # Combine parameters and well-being factors into a single record
                    record = {
                        "timepoint": timepoint,
                        "parameters": parameters,
                        "wellbeing_factors": wellbeing_factors,
                    }
                    json.dump(record, file, indent=4)
                    file.write("\n")

                    # Generate text using the sampled parameters
                    text_generator = TextGenerator(setup, config.prompts, parameters, generation_arguments)
                    generated_text = text_generator.out
                    json.dump({"generated_text": generated_text}, file, indent=4)
                    file.write("\n")