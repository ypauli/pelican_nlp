import torch

class Config:
    """
    A configuration class for setting up the model, generation parameters, and constants.
    """
    def __init__(self):
        self.model = "jphme/em_german_leo_mistral" #Athuin/tinyLama-german, meta-llama/Llama-3.2-1B, nikhilnagaraj/german_gpt_small, DiscoResearch/Llama3-German-8B
        self.parameters = {
            "prompts": [
                "Once upon a time", 
                "I am going to tell you about",
                "My favourite thing in the world is"
                ],
            "temperatures": [1.5, 2.5],
            "retroactive_spans": [20],
            "proactive_spans": [20],
            "num_beams": [2]
        }
        self.constants = {
            "calculate_metrics": True,
            "target_length": 200
        }
        
        self.file = "simu_results"
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")
