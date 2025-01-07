from datetime import datetime
import numpy as np

class Config:
    def __init__(self):
        self.subjects_per_cohort = 10
        self.timepoints_per_subject = 20
        self.cohorts = {
            "group_a": {
                "mean_values": [0.7, 0.8, 50, 50, 150, 0.15],  # temperature, sampling, retroactive_span, proactive_span, target_length, noise_rate
                "covariance_matrix": np.array([
                    [0.05,  0.03, -0.04, -0.04, -0.02,  0.03],  # temperature
                    [0.03,  0.04, -0.03, -0.03, -0.01,  0.02],  # sampling
                    [-0.04, -0.03, 10.0,   8.5,   5.0,  -0.02],  # retroactive_span
                    [-0.04, -0.03,  8.5,  10.0,   5.5,  -0.02],  # proactive_span
                    [-0.02, -0.01,  5.0,   5.5,  20.0,  -0.01],  # target_length
                    [0.03,  0.02, -0.02, -0.02, -0.01,  0.01]   # noise_rate
                ]),
                "std_devs": [0.1, 0.2, 3.0, 3.0, 5.0, 0.05]
            },
            "group_b": {
                "mean_values": [1.2, 1.5, 40, 40, 120, 0.2],
                "covariance_matrix": np.array([
                    [0.06,  0.04, -0.05, -0.05, -0.03,  0.04],
                    [0.04,  0.05, -0.04, -0.04, -0.02,  0.03],
                    [-0.05, -0.04, 12.0,  10.0,   6.0,  -0.03],
                    [-0.05, -0.04, 10.0,  12.0,   7.0,  -0.03],
                    [-0.03, -0.02,  6.0,   7.0,  18.0,  -0.02],
                    [0.04,  0.03, -0.03, -0.03, -0.02,  0.02]
                ]),
                "std_devs": [0.12, 0.25, 4.0, 4.0, 6.0, 0.06]
            },
            "group_c": {
                "mean_values": [0.5, 0.7, 60, 60, 180, 0.1], 
                "covariance_matrix": np.array([
                    [0.04,  0.02, -0.03, -0.03, -0.02,  0.02],
                    [0.02,  0.03, -0.02, -0.02, -0.01,  0.02],
                    [-0.03, -0.02,  8.0,   7.0,   4.5,  -0.02],
                    [-0.03, -0.02,  7.0,   8.0,   5.0,  -0.02],
                    [-0.02, -0.01,  4.5,   5.0,  15.0,  -0.01],
                    [0.02,  0.02, -0.02, -0.02, -0.01,  0.01]
                ]),
                "std_devs": [0.1, 0.15, 2.5, 2.5, 4.5, 0.03]
            },
            "group_d": {
                "mean_values": [1.5, 1.8, 30, 30, 100, 0.25], 
                "covariance_matrix": np.array([
                    [0.08,  0.06, -0.06, -0.06, -0.04,  0.05],
                    [0.06,  0.07, -0.05, -0.05, -0.03,  0.04],
                    [-0.06, -0.05, 15.0,  12.0,   8.0,  -0.04],
                    [-0.06, -0.05, 12.0,  15.0,   9.0,  -0.04],
                    [-0.04, -0.03,  8.0,   9.0,  25.0,  -0.03],
                    [0.05,  0.04, -0.04, -0.04, -0.03,  0.03]
                ]),
                "std_devs": [0.15, 0.3, 5.0, 5.0, 7.0, 0.07]
            }
        }
        self.parameter_rules = {
            "wellbeing": lambda parameters: round(
                0.4 * parameters["sleep"] + 0.4 * (1 - parameters["anxiety"]) + 0.2 * parameters["happiness"], 2
            ),
            "sleep": lambda parameters: round(
                1 - (parameters["retroactive_span"] - 30) / (200 - 30), 2
            ),
            "anxiety": lambda parameters: round(
                (200 - parameters["proactive_span"]) / (200 - 30), 2
            ),
            "happiness": lambda parameters: round(
                1 - (parameters["temperature"] - 0.8) / (5 - 0.8), 2
            ),
            "medication": lambda parameters: int(
                np.random.choice([0, 1], p=[(parameters["temperature"] - 0.8) / (5 - 0.8), 1 - (parameters["temperature"] - 0.8) / (5 - 0.8)])
            ),
            "drugs": lambda parameters: int(
                np.random.choice([0, 1], p=[1 - (parameters["temperature"] - 0.8) / (5 - 0.8), (parameters["temperature"] - 0.8) / (5 - 0.8)])
            ),
        }
        self.prompts = {
            "Seit letzter Woche hat sich in meinem Leben einiges verändert",
            "Mein letzter Traum war",
            "Von hier aus bis zum nächsten Supermarkt gelangt man am besten",  # "Der kürzeste Weg von hier aus zum nächsten Supermarkt ist",
            # "Dieses Bild zeigt", # change
            "Ich werde so viele Tiere aufzählen wie möglich: Pelikan,"
            # "Ich bin grundsätzlich zufrieden, hätte aber gerne", # change
            # "Ich wiederhole jetzt die Geschichte", # change
        }
        self.model_name = "jphme/em_german_leo_mistral"