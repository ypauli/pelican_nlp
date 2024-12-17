class ParameterGenerator:
    @staticmethod
    def generate_parameters(config, setup, options):
        """
        Calculate text generation parameters using adjustment rules from config.

        Args:
            config (Config): Config instance containing adjustment rules.
            options (dict): Generated wellbeing and binary values.

        Returns:
            dict: Adjusted text generation parameters.
        """
        parameters = {
            key: rule(options) for key, rule in config.parameter_rules.items()
        }
        generation_arguments = {
            "bad_words_ids": setup.excluded_tokens,
            "eos_token_id": None,
            "return_dict_in_generate": parameters["lie_rate"] > 0,
            "output_scores": parameters["lie_rate"] > 0,
            "use_cache": False,
            "temperature": parameters["temperature"],
            "num_beams": parameters["num_beams"], 
            "max_new_tokens": parameters["proactive_span"], 
            "do_sample": True,
            parameters["sampling"][0]: parameters["sampling"][1], # sampling_method: value
        }
        
        return parameters, generation_arguments
    