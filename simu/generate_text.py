import torch

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
            
            output_ids = setup.model.generate(
                input_ids,
                attention_mask = torch.ones_like(input_ids).to(device),
                **arguments
            )[:, -arguments["max_new_tokens"]:]
            
            
            text_ids = torch.cat((text_ids, output_ids), dim=1)    
            text += " " + setup.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # punctuation_mask = self.extract_punctuations(text_ids, setup.punctuation_tokens)
        print("output_ids length: ", text_ids.shape[1])
        
        return text
    
    # def extract_punctuations(self, text_ids, punctuation_tokens):
    #     punctuation_set = set(punctuation_tokens[0])
    #     punctuation_set.discard(1)
    #     punctuation_mask = [1 if token in punctuation_set else 0 for token in text_ids[0].tolist()]  
    #     return punctuation_mask
    
    def generate_arguments(self, setup, parameter):
        return {
            "bad_words_ids": setup.excluded_tokens,
            # "pad_token_id": setup.tokenizer.pad_token_id?,
            "eos_token_id": None,
            "output_scores": False, # returns logits
            "return_dict_in_generate": False,
            "use_cache": False,
            "temperature": parameter["temperature"],
            "do_sample": True,
            "num_beams": parameter["num_beams"], 
            "max_new_tokens": parameter["proactive_span"], 
            parameter["sampling"][0]: parameter["sampling"][1], # sampling_method: value
        }