import torch

class TextGenerator:
    def __init__(self, setup, prompts, parameter, generation_arguments):
        device = setup.device
        self.out = {}
        
        parameter["proactive_span"] = parameter["context_span"]
        parameter["retroactive_span"] = parameter["context_span"]
        
        for prompt in prompts:
            self.out[prompt] = self.generate_text(setup, device, prompt, parameter, generation_arguments)
    
    def generate_text(self, setup, device, text, parameter, arguments):
        text_ids = setup.tokenizer(text, return_tensors="pt").input_ids.to(device)

        while text_ids.shape[1] < parameter["target_length"]:
            input_ids = text_ids[:, -parameter["retroactive_span"]:]
            
            # Adjust amount of generated tokens if output becomes too long
            if text_ids.shape[1] + parameter["proactive_span"] > parameter["target_length"]:
                arguments["max_new_tokens"] = parameter["target_length"] - text_ids.shape[1] 
            
            output = setup.model.generate(
                input_ids,
                attention_mask = torch.ones_like(input_ids).to(device),
                **arguments
            )
            
            # Return only the newly generated tokens
            if isinstance(output, dict): 
                output_ids = output['sequences'][:, -arguments["max_new_tokens"]:]
            else:
                output_ids = output[:, -arguments["max_new_tokens"]:]            
                        
            # Concatenate results
            text_ids = torch.cat((text_ids, output_ids), dim=1)    
            text += " " + setup.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            text = text.replace("\n", "")
            
        return text