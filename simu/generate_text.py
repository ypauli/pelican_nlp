import torch
from torch.nn.functional import cosine_similarity
import random

class Generator:
    
    def __init__(self, setup, device, parameter, constants):
        arguments = self.generate_arguments(setup, parameter)
        if parameter["lie_rate"] > 0:
            self.token_embeddings = self.get_token_embeddings(setup)
        
        self.out = self.generate_text(setup, device, parameter, constants, arguments)
    
    def generate_text(self, setup, device, parameter, constants, arguments):
        text = parameter["prompt"]
        text_ids = setup.tokenizer(text, return_tensors="pt").input_ids.to(device)

        while text_ids.shape[1] < constants["target_length"]:
            input_ids = text_ids[:, -parameter["retroactive_span"]:]
            
            # Adjust amount of generated tokens if output becomes too long
            if text_ids.shape[1] + parameter["proactive_span"] > constants["target_length"]:
                arguments["max_new_tokens"] = constants["target_length"] - text_ids.shape[1] 
            
            # Introduce noise to input_ids
            input_ids = self.token_noise(input_ids, setup.tokenizer, parameter["token_noise_rate"])
            
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
            
            # Introduce lies into the generated text
            print("truthful output: ", setup.tokenizer.decode(output_ids[0], skip_special_tokens=True))
            
            if torch.rand(1).item() < parameter["lie_rate"]:
                lied_tokens = self.introduce_lies(setup, output_ids[:, :-1], parameter["truthfulness_penalty"])

                # Handle multi-token lies coherently
                if lied_tokens.shape[1] > 1:
                    output_ids[:, -lied_tokens.shape[1]:] = lied_tokens
                else:
                    output_ids[:, -1] = lied_tokens.squeeze(1) 
            
            print("output after lie: ", setup.tokenizer.decode(output_ids[0], skip_special_tokens=True))
                        
            # Concatenate results
            text_ids = torch.cat((text_ids, output_ids), dim=1)    
            text += " " + setup.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            text = text.replace("\n", "")
            
        return text
    
    def introduce_lies(self, setup, output_ids, truthfulness_penalty):
        if output_ids is None or output_ids.shape[0] == 0 or output_ids.shape[1] <= 1:
            print(f"Skipping introduce_lies due to invalid output_ids. Shape: {output_ids.shape}")
            return output_ids

        # Get logits for the last step
        logits = setup.model(input_ids=output_ids).logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)

        # Normalize embeddings
        context_embedding = self.get_context_embedding(output_ids, setup)
        token_embeddings = torch.nn.functional.normalize(self.token_embeddings.to(context_embedding.dtype), dim=-1)

        # Compute semantic similarity
        similarities = cosine_similarity(token_embeddings.cpu(), context_embedding.cpu().unsqueeze(0), dim=-1)
        similarities = torch.clamp(similarities, min=0, max=1)

        # Penalize probabilities
        penalized_probabilities = probabilities.clone()
        penalized_probabilities *= torch.clamp(1 - truthfulness_penalty * similarities.to(probabilities.device), min=0.1)

        # Normalize after penalization
        penalized_probabilities /= penalized_probabilities.sum(dim=-1, keepdim=True) + 1e-12

        # Add slight noise for diversity
        # noise = torch.rand_like(penalized_probabilities) * 0.05
        # penalized_probabilities += noise
        # penalized_probabilities /= penalized_probabilities.sum(dim=-1, keepdim=True)

        # Sample a new token
        new_tokens = torch.multinomial(penalized_probabilities, 1)

        return new_tokens

    def get_context_embedding(self, output_ids, setup):
        context_ids = output_ids[:, :-1]  # All tokens except the last generated one
        with torch.no_grad():
            context_outputs = setup.model(input_ids=context_ids, output_hidden_states=True)
            hidden_states = context_outputs.hidden_states[-1]        
        context_embedding = hidden_states[:, -1, :]  # Use the last hidden state
        context_embedding = context_embedding.mean(dim=0) + 1e-12  # Add epsilon for stability
        
        # Safeguards for numerical stability
        context_embedding = torch.clamp(context_embedding, min=-1e2, max=1e2)
        context_embedding = context_embedding.to(dtype=torch.float32)  # Ensure float32 precision
        context_embedding = torch.nn.functional.normalize(context_embedding, dim=-1)  # Normalize
        
        return context_embedding

    def get_token_embeddings(self, setup):
        with torch.no_grad():
            vocab_size = setup.tokenizer.vocab_size
            token_ids = torch.arange(vocab_size, device=setup.model.device).unsqueeze(0)
            token_embeddings = setup.model.get_input_embeddings()(token_ids).squeeze(0).cpu()  # Move to CPU
        return token_embeddings
    
    def token_noise(self, input_ids, tokenizer, noise_rate):
        if noise_rate < 0 or noise_rate > 1:
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
            "return_dict_in_generate": parameter["lie_rate"] > 0, # return logits if needed
            "output_scores": parameter["lie_rate"] > 0, # return logits if needed
            "use_cache": False,
            "temperature": parameter["temperature"],
            "num_beams": parameter["num_beams"], 
            "max_new_tokens": parameter["proactive_span"], 
            "do_sample": True,
            parameter["sampling"][0]: parameter["sampling"][1], # sampling_method: value
        }