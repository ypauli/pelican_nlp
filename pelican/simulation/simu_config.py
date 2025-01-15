import numpy as np
import math

class SimuConfig:
    def __init__(self):
        self.directory = "/home/ubuntu/PELICAN/pelican/simulation/simu_output"
        self.subjects = 2
        self.timepoints = 2
        self.global_parameter_stats = {
            "temperature": {"mean": 1.2, "variance": 0.4},
            "sampling": {"mean": 0.85, "variance": 0.01}, # Using top-p sampling
            "context_span": {"mean": 80, "variance": 650.79},
            "target_length": {"mean": 30, "variance": 0}, # {"mean": 120, "variance": 937} actual generation, {"mean": 30, "variance": 0} for test
        }
        self.cohorts = {
            "group_a": {
                "varied_parameter": "temperature",
                "mean_values": {"temperature": 1.2},
                "variance_values": {"temperature": 0.4},
            },
            "group_b": {
                "varied_parameter": "sampling",
                "mean_values": {"sampling": 0.85},
                "variance_values": {"sampling": 0.01},
            },
            "group_c": {
                "varied_parameter": "context_span",
                "mean_values": {"context_span": 80},
                "variance_values": {"context_span": 650.79},
            },
            # "group_d": {
            #     "varied_parameter": "target_length",
            #     "mean_values": {"target_length": 120},
            #     "variance_values": {"target_length": 937},
            # }
        }
        self.parameter_weights = {
            "temperature": -0.9,
            "sampling": -1.2,
            "context_span": 0.006,
            "target_length": 0.004,
        }
        self.parameter_rules = {
            # State score normalized between 0 and 1
            "state_score": lambda parameters: round(
                1 / (1 + math.exp(-sum(parameters[param] * weight for param, weight in self.parameter_weights.items()))),
                2
            ),
            # Wellbeing is directly equal to the state_score
            "wellbeing": lambda state_score: round(state_score, 2),
            # Sleep is highest when state_score is close to 0.75
            "sleep": lambda state_score: round(1 - abs(state_score - 0.75), 2),
            # Anxiety is highest when state_score is close to 0.25
            "anxiety": lambda state_score: round(1 - abs(state_score - 0.25), 2),
            # Happiness is the square the state_score
            "happiness": lambda state_score: round(state_score**2, 2),
            # Medication: More likely to be 1 if state_score is high
            "medication": lambda state_score: int(np.random.choice([0, 1], p=[1 - state_score, state_score])),
            # Drugs: Less likely than medication, and more likely to be 1 if state_score is low
            "drugs": lambda state_score: int(
                np.random.choice([0, 1], p=[
                    max(0, min(1, state_score + 0.3)),  # Ensure between 0 and 1
                    max(0, min(1, 0.7 - state_score))  # Ensure between 0 and 1
                ])
            )
        }
        self.prompts = {
            "Seit letzter Woche habe ich",
            "Als letztes habe ich geträumt",
            "Von hier aus bis zum nächsten Supermarkt gelangt man",
            "Ich werde so viele Tiere aufzählen wie möglich: Pelikan,"
            # Interactive Prompts
            # "Dieses Bild zeigt",
            # "Ich bin grundsätzlich zufrieden, hätte aber gerne", 
            # "Ich wiederhole jetzt die Geschichte",
        }
        self.model_name = "jphme/em_german_leo_mistral"