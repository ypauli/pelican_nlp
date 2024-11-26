import torch

class Generator:
    """
    A text generation class that supports token generation with constrained context and metric computation.
    """
    
    def __init__(self, setup, device, parameter, constants):
        """
        Initializes the Generator instance and generates the text with metrics.

        Args:
            setup (object): Object containing model and tokenizer setups.
            device (torch.device): The device to run the computations (e.g., 'cuda' or 'cpu').
            parameter (list): A list containing parameters for the generation process.
            constants (dict): A dictionary of constants required for text generation.
        """
        arguments = self.generate_arguments(setup, parameter, constants)
        
        if parameter[3][0] == -1: # if retroactive_span == -1 (no sliding window)
            self.out = self.generate_unbounded(setup, device, parameter, constants, arguments)
        else:
            self.out = self.generate_text(setup, device, parameter, constants, arguments)
        
    def generate_unbounded(self, setup, device, parameter, constants, arguments):
        text = parameter[0]
        text_ids = setup.tokenizer(text, return_tensors="pt").input_ids.to(device)
        metrics = [torch.empty((0,)).to(device) for _ in range(5)] #might have to adjust depending on which metrics are calculated
        output_ids = setup.model.generate(
                text_ids,
                attention_mask = torch.ones_like(text_ids).to(device),
                **arguments
            )
        
        if constants["calculate_metrics"]:
            logits = output_ids.scores
            output_sequences = output_ids.sequences[:, -arguments["max_new_tokens"]:]
            logits = torch.cat(
                [
                    logits[i][output_ids["beam_indices"][0, i], :].unsqueeze(0)
                    for i in range(len(logits))
                ],
                dim=0,
            ) 
            output_ids = output_ids.sequences[:, -arguments["max_new_tokens"]:] 
            metrics = self.calculate_metrics(output_sequences, logits)
        else:
            output_ids = output_ids[:, -arguments["max_new_tokens"]:] 
                
        text_ids = torch.cat((text_ids, output_ids), dim=1)    
        output = setup.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        text += " " + output
        # print("text: ", text)
            
        return text, metrics 
    
    def generate_text(self, setup, device, parameter, constants, arguments):
        """
        Generates text based on input parameters and calculates metrics.

        Args:
            setup (object): Object containing model and tokenizer setups.
            device (torch.device): The device to run the computations (e.g., 'cuda' or 'cpu').
            parameter (list): A list containing parameters for the generation process.
            constants (dict): A dictionary of constants required for text generation.
            arguments (dict): A dictionary of arguments passed to the generation model.

        Returns:
            tuple: Generated text (str) and list of accumulated metrics (list of torch.Tensor).
        """
        prompt = parameter[0]
        retroactive_span = parameter[3][0]
        proactive_span = parameter[3][1]
        
        text = prompt
        text_ids = setup.tokenizer(text, return_tensors="pt").input_ids.to(device)
        metrics = [torch.empty((0,)).to(device) for _ in range(4)] #might have to adjust depending on which metrics are calculated

        while text_ids.shape[1] < constants["target_length"]:
            
            input_ids = text_ids[:, -retroactive_span:]
            output_ids = setup.model.generate(
                input_ids,
                attention_mask = torch.ones_like(input_ids).to(device),
                **arguments
            )
            
            if constants["calculate_metrics"]:
                logits = output_ids.scores
                output_sequences = output_ids.sequences[:, -proactive_span:]
                
                logits = torch.cat(
                    [
                        logits[i][output_ids["beam_indices"][0, i], :].unsqueeze(0)
                        for i in range(len(logits))
                    ],
                    dim=0,
                )              
                               
                output_ids = output_ids.sequences[:, -proactive_span:] 
                
                current_metrics = self.calculate_metrics(output_sequences, logits)
                metrics = self.concatenate_metrics(metrics, current_metrics)
            else:
                output_ids = output_ids[:, -proactive_span:] 
            
            text_ids = torch.cat((text_ids, output_ids), dim=1)    
            output = setup.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            text += " " + output
            # print("text: ", text)
            
        return text, metrics
    
    def generate_arguments(self, setup, parameter, constants):
        """
        Prepares a dictionary of arguments for the generation process.

        Args:
            setup (object): Object containing model and tokenizer setups.
            parameter (list): A list containing parameters for the generation process.
            constants (dict): A dictionary of constants required for text generation.

        Returns:
            dict: Arguments to be used for the model generation.
        """
        return {
            # Set parameters, constant for all generations
            # "bad_words_ids": setup.excluded_tokens,
            "pad_token_id": setup.tokenizer.pad_token_id,
            "output_scores": constants["calculate_metrics"], # returns logits
            "return_dict_in_generate": constants["calculate_metrics"],
            "use_cache": False,
            "temperature": parameter[1],
            "do_sample": True,
            parameter[4][0]: parameter[4][1], # sampling_method: value
            "num_beams": parameter[2],
            "max_new_tokens": parameter[3][1], # proactive_span
        }
     
    def calculate_metrics(self, output_only, logits_tensor): 
        """
        Computes various metrics for generated sequences.

        Args:
            output_only (torch.Tensor): Output token sequences from the model.
            logits_tensor (torch.Tensor): Logits corresponding to the output tokens.

        Returns:
            tuple: A tuple containing metrics:
                - probability differences (torch.Tensor)
                - entropy (torch.Tensor)
                - information content (torch.Tensor)
                - entropy deviations (torch.Tensor)
                - log probabilities (torch.Tensor)
        """
        probs = torch.nn.functional.softmax(logits_tensor, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        actual_probs = torch.gather(probs, 1, output_only.t()).t().squeeze()
        # log_actual_probs = torch.log(actual_probs +1e-9)

        probability_differences_tensor_single_generation = max_probs - actual_probs
        entropy_tensor_single_generation = (-probs * log_probs).sum(dim=-1)
        information_content_tensor_single_generation = -torch.log(actual_probs + 1e-9)
        entropy_deviations_tensor_single_generation = (entropy_tensor_single_generation - information_content_tensor_single_generation)

        #precalculations for perplexity
        # log_prob_tens = torch.tensor(log_actual_probs, device="cuda")
        # finite_mask = torch.isfinite(log_prob_tens)
        # finite_mean = log_prob_tens[finite_mask].mean()
        # log_prob_tens = torch.where(finite_mask, log_prob_tens, finite_mean)

        return (
            probability_differences_tensor_single_generation,
            entropy_tensor_single_generation,
            information_content_tensor_single_generation,
            entropy_deviations_tensor_single_generation,
            # log_prob_tens
        ) 
     
    def concatenate_metrics(self, metrics, current_metrics):
        """
        Concatenates the current batch of metrics with the accumulated metrics.

        Args:
            metrics (list of torch.Tensor): List of tensors to store accumulated metrics.
            current_metrics (tuple of torch.Tensor): Tensors representing the current batch of metrics.

        Returns:
            list of torch.Tensor: Updated metrics with concatenated values.
        """
        # Ensure metrics and current_metrics have the same number of elements
        if len(metrics) != len(current_metrics):
            raise ValueError("Mismatch between metrics and current_metrics length.")
        for i in range(len(metrics)):
            metrics[i] = torch.cat((metrics[i], current_metrics[i]), dim=0)
        
        return metrics