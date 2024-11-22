from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
from transformers import AutoModelForCausalLM, init_empty_weights

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_instance = None
        self.device_map = None

    def load_model(self):
        self.model_instantiation()
        self.device_map_creation()
        self.model_instance = dispatch_model(self.model_instance, device_map=self.device_map)

    def model_instantiation(self,empty_weights=False):
        if empty_weights:
            with init_empty_weights():
                self.model_instance = AutoModelForCausalLM.from_pretrained(self.model_name)
        else:
            self.model_instance = AutoModelForCausalLM.from_pretrained(self.model_name)
        return self.model_instance

    def device_map_creation(self):
        self.device_map = infer_auto_device_map(self.model_instance, max_memory={
            0: self.config.VRAM_str,
            'cpu': self.config.RAM_str,
            'disk': '200GB'
        })
        return self.device_map