from datetime import datetime
import numpy as np
import math

    # parameters = {
    #     "temperature": round(np.clip(sampled_values[0], 0.8, 5.0), 2),
    #     "sampling": ("top_p", round(np.clip(sampled_values[1], 0.75, 2.0), 2)),
    #     "retroactive_span": int(np.clip(sampled_values[2], 30, 300)),
    #     "proactive_span": int(np.clip(sampled_values[3], 30, 300)),
    #     "target_length": int(np.clip(sampled_values[4], 150, 450)), # roughly 100 - 350 words x 1.5 to account for tokens
    #     "token_noise_rate": round(np.clip(sampled_values[5], 0.0, 0.2), 2),
    #     "lie_rate": 0,

class Config:
    def __init__(self):
        self.subjects_per_cohort = 10
        self.timepoints_per_subject = 5
        self.global_parameter_stats = {
            "temperature": {"mean": 1.2, "variance": 0.0234},
            "sampling": {"mean": 0.85, "variance": 0.0026}, # Using top-p sampling
            "context_span": {"mean": 150, "variance": 2108.57},
            "target_length": {"mean": 50, "variance": 26.03}, # use 300, 1275.56 in actual generation, "mean": 50, "variance": 26.03 for test
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
                "mean_values": {"context_span": 150},
                "variance_values": {"context_span": 3149.85},
            },
            "group_d": {
                "varied_parameter": "target_length",
                "mean_values": {"target_length": 300},
                "variance_values": {"target_length": 8434.30},
            }
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
            "Seit letzter Woche hat sich in meinem Leben einiges verändert", # remove before generation
            "Seit letzter Woche habe ich",
            "Mein letzter Traum war",
            "Als letztes habe ich geträumt, dass" # remove before generation
            "Von hier aus bis zum nächsten Supermarkt gelangt man am besten",
            "Ich werde so viele Tiere aufzählen wie möglich: Pelikan,"
            # Interactive Prompts
            # "Dieses Bild zeigt",
            # "Ich bin grundsätzlich zufrieden, hätte aber gerne", 
            # "Ich wiederhole jetzt die Geschichte",
        }
        self.model_name = "jphme/em_german_leo_mistral"