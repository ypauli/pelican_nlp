import os.path

from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
from transformers import AutoModelForCausalLM
import torch
import psutil

class Model:
    def __init__(self, model_name, project_path):
        self.model_name = model_name
        self.model_instance = None
        self.device_map = None
        self.PROJECT_PATH = project_path

    def load_model(self):
        self.model_instantiation()
        self.device_map_creation()
        offload_directory = os.path.join(self.PROJECT_PATH, 'offload')
        try:
            os.makedirs(offload_directory, exist_ok=True)
        except Exception as e:
            print(f"Error: {e}")
        self.model_instance = dispatch_model(self.model_instance, device_map=self.device_map, offload_dir=offload_directory)

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