import numpy as np

class ParameterGenerator:
    @staticmethod
    def generate_parameters(config, setup, cohort_name):
        """
        Sample generation parameters for a given cohort.
        
        Args:
            config (Config): Configuration instance containing cohort settings.
            cohort_name (str): Name of the cohort to generate parameters for.

        Returns:
            dict: A dictionary of sampled generation parameters.
        """
        # Retrieve cohort-specific configuration
        cohort = config.cohorts[cohort_name]

        # Generate sampled parameters using a multivariate normal distribution
        sampled_values = np.random.multivariate_normal(
            mean=cohort["mean_values"],
            cov=np.outer(cohort["std_devs"], cohort["std_devs"]) * cohort["covariance_matrix"]
        )

        # Clip values to ensure they stay within acceptable ranges
        parameters = {
            "temperature": round(np.clip(sampled_values[0], 0.8, 5.0), 2),
            "sampling": ("top_p", round(np.clip(sampled_values[1], 0.8, 2.0), 2)),
            "retroactive_span": int(np.clip(sampled_values[2], 30, 200)),
            "proactive_span": int(np.clip(sampled_values[3], 30, 200)),
            "target_length": int(np.clip(sampled_values[4], 150, 450)), # roughly 100 - 350 words x 1.5 to account for tokens
            "token_noise_rate": round(np.clip(sampled_values[5], 0.0, 0.3), 2),
            "lie_rate": 0,
        }
        
        generation_arguments = {
            "bad_words_ids": setup.excluded_tokens,  # Excluded tokens
            "eos_token_id": None,                    # No specific end-of-sequence token
            "return_dict_in_generate": parameters.get("lie_rate", 0) > 0,  # Include logits if lies are introduced
            "output_scores": parameters.get("lie_rate", 0) > 0,            # Include logits for scoring
            "use_cache": False,                      # Disable cache for dynamic text generation
            "temperature": parameters["temperature"],  # Sampling temperature
            "num_beams": 5,                           # Fixed number of beams
            "max_new_tokens": parameters["proactive_span"],  # Maximum tokens to generate in one step
            "do_sample": True,                        # Enable sampling
            parameters["sampling"][0]: parameters["sampling"][1],  # Sampling method (e.g., "top_p")
        }

        return parameters, generation_arguments