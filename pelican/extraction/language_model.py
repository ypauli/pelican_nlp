import torch
import psutil

from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
from transformers import AutoModelForCausalLM

class Model:
    def __init__(self, model_name, project_path):
        self.model_name = model_name
        self.model_instance = None
        self.device_map = None
        self.PROJECT_PATH = project_path

    def load_model(self):
        """Loads and configures the model"""

        if self.model_name == 'fastText':
            import fasttext
            fasttext.util.download_model('de', if_exists='ignore')
            self.model_instance = fasttext.load_model('cc.de.300.bin')
            print('FastText model loaded.')
        elif self.model_name == 'xlm-roberta-base':
            from transformers import AutoModel
            self.model_instance = AutoModel.from_pretrained(self.model_name)
            print('RoBERTa model loaded.')
        else:
            raise ValueError("Invalid model name.")

        # Additional model setup
        self.device_map_creation()

        self.model_instance = dispatch_model(self.model_instance, device_map=self.device_map)
        print('Model dispatched to appropriate devices.')

    def model_instantiation(self,empty_weights=False):
        if empty_weights:
            with init_empty_weights():
                self.model_instance = AutoModelForCausalLM.from_pretrained(self.model_name)
        else:
            self.model_instance = AutoModelForCausalLM.from_pretrained(self.model_name)

    def device_map_creation(self):
        #check if cuda is available
        if not torch.cuda.is_available():
            print('Careful: Cuda not available, using CPU. This will be very slow.')
        else:
            print(f'{torch.cuda.get_device_name(0)} available.')

        available_VRAM = str(int(torch.cuda.get_device_properties(0).total_memory/(1024 ** 3))-3)+'GB'
        available_RAM = str(int(psutil.virtual_memory().total/(1024 ** 3))-3)+'GB'

        #create device map and offload directory if it doesn't exist
        self.device_map = infer_auto_device_map(self.model_instance, max_memory={
            0: available_VRAM,
            'cpu': available_RAM,
            'disk': '200GB'
        })