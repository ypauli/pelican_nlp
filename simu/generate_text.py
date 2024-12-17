import torch
from torch.nn.functional import cosine_similarity


class TextGenerator:
    def __init__(self, setup, parameter, arguments):
        """Initialize the TextGenerator with shared settings."""
        self.setup = setup
        self.parameter = parameter
        self.arguments = arguments
        self.token_embeddings = self._get_token_embeddings()

    def generate_text(self, prompts):
        """
        Generate text for a list of prompts.

        Args:
            prompts (list): List of input prompts for text generation.

        Returns:
            list: A list of generated outputs corresponding to each prompt.
        """
        results = {}
        for prompt in prompts:
            results[prompt] = self._generate_single_text(prompt)
        return results

    def _generate_single_text(self, text):
        """
        Core text generation logic for a single prompt.

        Args:
            text (str): The input prompt for text generation.

        Returns:
            str: The generated text output.
        """
        device = self.setup.device
        target_length = self.parameter["target_length"]
        proactive_span = self.parameter["proactive_span"]
        text_ids = self.setup.tokenizer(text, return_tensors="pt").input_ids.to(device)
        current_length = text_ids.shape[1]

        while current_length < target_length:
            input_ids = text_ids[:, -self.parameter["retroactive_span"]:]
            self._adjust_arguments(current_length, target_length)

            # Introduce token noise
            input_ids = self._token_noise(input_ids, self.setup.tokenizer, self.parameter["token_noise_rate"])

            # Generate new tokens
            output_ids = self._generate_tokens(input_ids)

            # Introduce lies
            output_ids = self._introduce_lies(output_ids)

            # Update text and input_ids
            text_ids = torch.cat((text_ids, output_ids), dim=1)
            new_text = self.setup.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            text += " " + new_text

            current_length = text_ids.shape[1]

        return self._clean_text(text)

    def _adjust_arguments(self, current_length, target_length):
        """Adjust arguments for token generation."""
        proactive_span = self.parameter["proactive_span"]
        if current_length + proactive_span > target_length:
            self.arguments["max_new_tokens"] = target_length - current_length

    def _generate_tokens(self, input_ids):
        """Generate new tokens using the model."""
        output = self.setup.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids).to(input_ids.device),
            **self.arguments
        )
        if isinstance(output, dict):
            return output['sequences'][:, -self.arguments["max_new_tokens"]:]
        return output[:, -self.arguments["max_new_tokens"]:]

    def _introduce_lies(self, output_ids):
        """
        Introduce false tokens based on semantic penalties.

        Args:
            output_ids (Tensor): The current output token IDs.

        Returns:
            Tensor: Modified output token IDs.
        """
        lie_rate = self.parameter.get("lie_rate", 0.0)
        truthfulness_penalty = self.parameter.get("truthfulness_penalty", 0.0)

        if lie_rate < torch.rand(1).item():
            return output_ids  # Skip introducing lies

        logits = self.setup.model(input_ids=output_ids).logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        context_embedding = self._get_context_embedding(output_ids)
        token_embeddings = torch.nn.functional.normalize(self.token_embeddings, dim=-1)

        # Compute similarity between tokens and context
        similarities = cosine_similarity(token_embeddings.cpu(), context_embedding.cpu().unsqueeze(0), dim=-1)
        similarities = torch.clamp(similarities, min=0, max=1)

        # Penalize probabilities based on similarity
        penalized_probabilities = probabilities.clone()
        penalized_probabilities *= torch.clamp(1 - truthfulness_penalty * similarities.to(probabilities.device), min=0.1)
        penalized_probabilities /= penalized_probabilities.sum(dim=-1, keepdim=True) + 1e-12

        # Sample a new token
        new_tokens = torch.multinomial(penalized_probabilities, 1)
        output_ids[:, -1] = new_tokens.squeeze(1)
        return output_ids

    def _get_context_embedding(self, output_ids):
        """
        Compute the normalized context embedding for the current sequence.

        Args:
            output_ids (Tensor): Token IDs for the sequence.

        Returns:
            Tensor: The context embedding vector.
        """
        context_ids = output_ids[:, :-1]
        with torch.no_grad():
            hidden_states = self.setup.model(input_ids=context_ids, output_hidden_states=True).hidden_states[-1]
        context_embedding = hidden_states[:, -1, :].mean(dim=0) + 1e-12
        return torch.nn.functional.normalize(context_embedding, dim=-1)

    def _get_token_embeddings(self):
        """
        Retrieve token embeddings from the model.

        Returns:
            Tensor: Token embeddings for the entire vocabulary.
        """
        with torch.no_grad():
            vocab_size = self.setup.tokenizer.vocab_size
            token_ids = torch.arange(vocab_size, device=self.setup.model.device).unsqueeze(0)
            return self.setup.model.get_input_embeddings()(token_ids).squeeze(0).cpu()

    def _clean_text(self, text):
        """Clean text to remove unwanted characters."""
        return text.replace("\n", "")

    def _token_noise(self, input_ids, tokenizer, noise_rate):
        """Introduce random token noise into the input IDs."""
        if noise_rate <= 0 or noise_rate > 1:
            return input_ids

        batch_size, seq_length = input_ids.shape
        noise_mask = torch.rand((batch_size, seq_length), device=input_ids.device) < noise_rate
        random_tokens = torch.randint(
            low=0,
            high=tokenizer.vocab_size,
            size=(batch_size, seq_length),
            device=input_ids.device
        )
        return torch.where(noise_mask, random_tokens, input_ids)