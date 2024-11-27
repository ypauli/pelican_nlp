import torch
from datetime import datetime

class Config:
    """
    A configuration class for setting up the model, generation parameters, and constants.
    """
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")
        self.model = "jphme/em_german_leo_mistral" #Athuin/tinyLama-german, meta-llama/Llama-3.2-1B, nikhilnagaraj/german_gpt_small, DiscoResearch/Llama3-German-8B
        
        self.parameters = {
            "prompts": [
                "Es war einmal",
                "Ich erzähle ihnen jetzt eine Geschichte",
                "Mein grösstes Hobby ist"
                ],
            "temperatures": [1.5, 2.5],
            "num_beams": [2],
            
            "retroactive_spans": [100, 20], # choose retroactive_span = target_length for unbounded context
            "proactive_spans": [20, 50],
            
            "sampling": {
                "top_p": [0.2, 0.6, 1.0],
                "top_k": [4, 16, 32, 64],
                "typical_p": [0.2, 0.6, 1.0]
            } ,
        }
       
        self.constants = {
            "calculate_metrics": True,
            "target_length": 100 # including the length of the prompt
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        self.file = f"output_{timestamp}.json"
