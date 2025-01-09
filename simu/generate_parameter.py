import numpy as np

class ParameterGenerator:
    @staticmethod
    def subject_sample(config, cohort_name):
        """
        Set up subject-specific parameters, including constant values
        and the mean/variance for the cohort's varied parameter.
        
        Args:
            config (Config): The configuration object.
            cohort_name (str): The name of the cohort.
        
        Returns:
            dict: Subject-specific constants and cohort-specific varying parameter details.
        """
        # Cohort-specific configuration
        cohort = config.cohorts[cohort_name]

        # Sample constants for non-varying parameters
        constants = {
            param: np.random.normal(
                loc=config.global_parameter_stats[param]["mean"],
                scale=np.sqrt(config.global_parameter_stats[param]["variance"])
            )
            for param in config.global_parameter_stats
            if param != cohort["varied_parameter"]  # Exclude the varied parameter
        }

        # Sample mean and variance for the cohort's varied parameter
        varied_param = cohort["varied_parameter"]
        varied_param_mean = np.random.normal(
            loc=cohort["mean_values"][varied_param],
            scale=np.sqrt(cohort["variance_values"][varied_param])
        )
        varied_param_variance = np.abs(np.random.normal(
            loc=cohort["variance_values"][varied_param],
            scale=0.1 * cohort["variance_values"][varied_param]
        ))  # Ensure variance is positive

        return {
            "constants": constants,
            "varied_parameter": varied_param,
            "varied_param_mean": varied_param_mean,
            "varied_param_variance": varied_param_variance,
        }
        
    @staticmethod
    def timepoint_sample(varied_param_range):
        """
        Generate a timepoint-specific value for the varied parameter.

        Args:
            varied_param_mean (float): The mean value for the varied parameter.
            varied_param_variance (float): The variance for the varied parameter.

        Returns:
            float: Timepoint-specific value for the varied parameter.
        """
        return np.random.uniform(varied_param_range)

    @staticmethod
    def wellbeing_sample(parameters, parameter_rules):
        """
        Calculate well-being factors based on parameters and rules.

        Args:
            parameters (dict): A dictionary of subject and timepoint-specific parameters.
            parameter_rules (dict): A dictionary of lambda functions defining well-being rules.

        Returns:
            dict: Calculated well-being factors.
        """
        return {
            key: rule(parameters) for key, rule in parameter_rules.items()
        }
        
    @staticmethod
    def generate_parameters(parameters, setup):
        """
        Generate generation arguments for TextGenerator based on the parameters.

        Args:
            parameters (dict): Dictionary of current parameters (subject-specific and timepoint-specific).
            setup (PipelineSetup): The pipeline setup object for accessing excluded tokens, etc.

        Returns:
            dict: Generation arguments for TextGenerator.
        """
        return {
            "bad_words_ids": setup.excluded_tokens,  # Excluded tokens
            "eos_token_id": None,                    # No specific end-of-sequence token
            "return_dict_in_generate": parameters.get("lie_rate", 0) > 0,  # Include logits if lies are introduced
            "output_scores": parameters.get("lie_rate", 0) > 0,            # Include logits for scoring
            "use_cache": False,                      # Disable cache for dynamic text generation
            "temperature": parameters["temperature"],  # Sampling temperature
            "num_beams": 5,                           # Fixed number of beams
            "max_new_tokens": parameters["context_span"],  # Maximum tokens to generate in one step, proactive_span
            "do_sample": True,                        # Enable sampling
            "top_p": parameters["sampling"],  # Sampling method (e.g., "top_p")
        }