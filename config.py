from datetime import datetime

class Config:
    def __init__(self):
        self.num_subjects = 10
        self.num_samples = 5
        self.inter_subject_variance = 0.5
        self.intra_subject_variance = 0.2
        self.prompts = {
            "Seit letzter Woche hat sich in meinem Leben einiges verändert",
            "Mein letzter Traum war",
            "Der kürzeste Weg von hier aus zum nächsten Supermarkt ist",
            "Ich werde so viele Tiere aufzählen wie möglich",
        }
        self.model_name = "jphme/em_german_leo_mistral"
        self.output_format = f"_{datetime.now().strftime("%Y%m%d_%H%M%S%f")}"