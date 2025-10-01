import torch
import psutil
import os
import shutil

from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel

class Model:
    def __init__(self, model_name, project_path):
        self.model_name = model_name
        self.model_instance = None
        self.device_map = None
        self.PROJECT_PATH = project_path

    def load_model(self, empty_weights=False, trust_remote_code=False):
        """Loads and configures the model"""

        if self.model_name == 'fastText':
            import fasttext
            import fasttext.util
            
            # Create a model directory if it doesn't exist
            model_dir = os.path.join(os.path.expanduser('~'), '.fasttext')
            os.makedirs(model_dir, exist_ok=True)
            
            # Set the model path using proper OS path joining
            model_path = os.path.join(model_dir, 'cc.de.300.bin')
            
            # Download only if model doesn't exist or is invalid
            need_download = True
            if os.path.exists(model_path):
                try:
                    self.model_instance = fasttext.load_model(model_path)
                    need_download = False
                except ValueError:
                    print(f"Existing model file is corrupted, re-downloading...")
                    os.remove(model_path)
            
            if need_download:
                print("Downloading FastText model...")
                try:
                    # Try the built-in FastText downloader first
                    fasttext.util.download_model('de', if_exists='ignore')
                    # Find the downloaded file in current directory
                    downloaded_file = 'cc.de.300.bin'
                    if os.path.exists(downloaded_file):
                        # Move the file to the correct location
                        shutil.move(downloaded_file, model_path)
                    else:
                        raise FileNotFoundError("FastText downloader didn't create the expected file")
                except (OSError, ValueError, FileNotFoundError) as e:
                    print(f"FastText downloader failed, using direct download: {str(e)}")
                    # Direct download fallback
                    import urllib.request
                    url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz'
                    print(f"Downloading from {url}...")
                    temp_gz_path = model_path + '.gz'
                    urllib.request.urlretrieve(url, temp_gz_path)
                    
                    # Decompress the file
                    print("Decompressing model file...")
                    import gzip
                    with gzip.open(temp_gz_path, 'rb') as f_in:
                        with open(model_path, 'wb') as f_out:
                            f_out.write(f_in.read())
                    os.remove(temp_gz_path)
                    print("Model decompressed successfully")
                
                # Verify the downloaded model
                try:
                    self.model_instance = fasttext.load_model(model_path)
                except ValueError as e:
                    raise ValueError(f"Failed to load downloaded model: {str(e)}. Please try removing {model_path} and running again.")
            
            print(f'FastText model loaded successfully from {model_path}')
        elif self.model_name == 'xlm-roberta-base':
            self.model_instance = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=trust_remote_code,
                use_safetensors=True
            )
            print('RoBERTa model loaded.')
        elif self.model_name == 'DiscoResearch/Llama3-German-8B-32k':
            if empty_weights:
                with init_empty_weights():
                    self.model_instance = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        trust_remote_code=trust_remote_code,
                        use_safetensors=True
                    )
            else:
                self.model_instance = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=trust_remote_code,
                    use_safetensors=True
                )
            print(f'Llama3-German-8B-32k loaded')
        else:
            raise ValueError("Invalid model name.")

        if self.model_name == 'xlm-roberta-base' or self.model_name == 'DiscoResearch/Llama3-German-8B-32k':
            # Additional model setup
            self.device_map_creation()

            self.model_instance = dispatch_model(self.model_instance, device_map=self.device_map)
            print('Model dispatched to appropriate devices.')

    def device_map_creation(self):
        # Detect device
        if torch.cuda.is_available():
            device_type = "cuda"
            print(f'{torch.cuda.get_device_name(0)} available.')
            available_VRAM = str(int(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)) - 3) + 'GB'
            max_memory = {
                0: available_VRAM,
                "cpu": str(int(psutil.virtual_memory().total / (1024 ** 3)) - 3) + "GB",
                "disk": "200GB",
            }

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
            print("Apple Metal (MPS) available.")
            # MPS doesn’t expose VRAM the same way, so just use CPU + MPS fallback
            max_memory = {
                "mps": str(int(psutil.virtual_memory().total / (1024 ** 3)) - 3) + "GB",
                "cpu": str(int(psutil.virtual_memory().total / (1024 ** 3)) - 3) + "GB",
                "disk": "200GB",
            }

        else:
            device_type = "cpu"
            print("Careful: No GPU available, using CPU. This will be slow.")
            max_memory = {
                "cpu": str(int(psutil.virtual_memory().total / (1024 ** 3)) - 3) + "GB",
                "disk": "200GB",
            }

        # Create device map
        self.device_map = infer_auto_device_map(self.model_instance, max_memory=max_memory)
        return device_type