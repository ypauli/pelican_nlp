import numpy as np

class OptionGenerator:
    @staticmethod
    def generate_options(timepoint, config, covariances):
        """
        Sample continuous and binary wellbeing factors.

        Parameters:
        - mean_values: List of mean values for continuous variables.
        - covariance_matrix: Covariance matrix for continuous variables.
        - binary_probs: Tuple of probabilities for binary variables (medication, drugs).

        Returns:
        - A dictionary containing sampled values.
        """
        
        mean_values = config.mean_values
        binary_probs = config.binary_probs
        
        continuous_sample = np.random.multivariate_normal(mean_values, covariances)
        continuous_sample = np.clip(continuous_sample, 0, 1)
        continuous_sample = [round(float(val), 2) for val in continuous_sample]

        medication = int(np.random.choice([0, 1], p=[1 - binary_probs[0], binary_probs[0]]))
        drugs = int(np.random.choice([0, 1], p=[1 - binary_probs[1], binary_probs[1]]))

        return {
            "timepoint": timepoint,
            "wellbeing": float(continuous_sample[0]),
            "sleep": float(continuous_sample[1]),
            "happiness": float(continuous_sample[2]),
            "anxiety": float(continuous_sample[3]),
            "medication": medication,
            "drugs": drugs
        }
