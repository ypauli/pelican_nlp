import torch

class TextGenerator:
    """
    A class for generating text using a language model, given initial prompts, parameters, 
    and generation arguments.

    Attributes:
        out (dict): A dictionary mapping each input prompt to its generated output text.
    """
    
    def __init__(self, setup, prompts, parameter, generation_arguments):
        """
        Initialize the TextGenerator and generate text for each prompt.

        Args:
            setup (object): An object containing the model, tokenizer, and device information.
            prompts (list): A list of initial text prompts to generate text from.
            parameter (dict): A dictionary of generation parameters, including `context_span`, 
                              `target_length`, `proactive_span`, and `retroactive_span`.
            generation_arguments (dict): A dictionary of arguments to control the text generation process.

        Attributes:
            out (dict): A dictionary storing the generated text for each input prompt.
        """
        device = setup.device
        self.out = {}
        
        parameter["proactive_span"] = parameter["context_span"]
        parameter["retroactive_span"] = parameter["context_span"]
        
        for prompt in prompts:
            self.out[prompt] = self.generate_text(setup, device, prompt, parameter, generation_arguments)
    
    def generate_text(self, setup, device, prompt, parameter, arguments):
        """
        Generate text for a single prompt based on specified parameters and arguments.

        Args:
            setup (object): An object containing the model, tokenizer, and device information.
            device (torch.device): The device to run the text generation (e.g., GPU or CPU).
            prompt (str): The initial text prompt to start generating text from.
            parameter (dict): A dictionary of generation parameters, including `target_length`, 
                              `proactive_span`, and `retroactive_span`.
            arguments (dict): A dictionary of arguments for the text generation process, 
                              such as sampling temperature, max_new_tokens, and others.

        Returns:
            str: The generated text based on the input prompt and parameters.
        """
        text = prompt
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