import torch
import random

class Generator:
    
    def __init__(self, setup, device, parameter, constants):
        arguments = self.generate_arguments(setup, parameter)
        self.out = self.generate_text(setup, device, parameter, constants, arguments)
    
    def generate_text(self, setup, device, parameter, constants, arguments):
        text = parameter["prompt"]
        text_ids = setup.tokenizer(text, return_tensors="pt").input_ids.to(device)

        while text_ids.shape[1] < constants["target_length"]:
            input_ids = text_ids[:, -parameter["retroactive_span"]:]
            
            if text_ids.shape[1] + parameter["proactive_span"] > constants["target_length"]:
                arguments["max_new_tokens"] = constants["target_length"] - text_ids.shape[1] # adjust amount of generated tokens if output becomes too long
            
            # Introduce noise to input_ids
            print("before: ", input_ids)
            input_ids = self.token_noise(input_ids, setup.tokenizer, parameter["token_noise_rate"])
            print("after: ", input_ids)
            
            output_ids = setup.model.generate(
                input_ids,
                attention_mask = torch.ones_like(input_ids).to(device),
                **arguments
            )[:, -arguments["max_new_tokens"]:]
            
            text_ids = torch.cat((text_ids, output_ids), dim=1)    
            text += " " + setup.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            text = text.replace("\n", "")
            
        return text
    
    def token_noise(self, input_ids, tokenizer, noise_rate):
        if noise_rate <= 0 or noise_rate > 1:
            raise ValueError("noise_rate must be in the range (0, 1].")
        
        batch_size, seq_length = input_ids.shape
        num_noisy_tokens = int(seq_length * noise_rate)
        if num_noisy_tokens == 0:
            return input_ids

        noise_mask = torch.rand((batch_size, seq_length), device=input_ids.device) < (noise_rate)
        random_tokens = torch.randint(
            low=0,
            high=tokenizer.vocab_size,
            size=(batch_size, seq_length),
            device=input_ids.device
        )
        noisy_input_ids = torch.where(noise_mask, random_tokens, input_ids)
        return noisy_input_ids
    
    def generate_arguments(self, setup, parameter):
        return {
            "bad_words_ids": setup.excluded_tokens,
            # "pad_token_id": setup.tokenizer.pad_token_id?,
            "eos_token_id": None,
            "output_scores": False, # returns logits
            "return_dict_in_generate": False,
            "use_cache": False,
            "temperature": parameter["temperature"],
            "num_beams": parameter["num_beams"], 
            "max_new_tokens": parameter["proactive_span"], 
            "do_sample": True,
            parameter["sampling"][0]: parameter["sampling"][1], # sampling_method: value
        }