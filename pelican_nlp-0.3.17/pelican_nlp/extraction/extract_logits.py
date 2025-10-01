import torch
import torch.nn.functional as F
from tqdm import tqdm

from pelican_nlp.config import debug_print

class LogitsExtractor:
    def __init__(self, options, pipeline, project_path):

        self.device = 'cuda' if torch.cuda.is_available()==True else 'cpu'
        self.options = options
        self.model_name = self.options['model_name']
        self.pipeline = pipeline
        self.PROJECT_PATH = project_path

    def extract_features(self, section, tokenizer, model):

        debug_print(f'section to tokenize: {section}')
        tokens = tokenizer.tokenize_text(section)
        debug_print(tokens)

        chunk_size = self.options['chunk_size']
        overlap_size = self.options['overlap_size']

        # Convert list of token IDs to tensor if needed
        if isinstance(tokens, list):
            input_ids = torch.tensor([tokens], device=self.device)
        else:
            input_ids = tokens.to(self.device)
            
        chunks = self._split_into_chunks(input_ids, chunk_size, overlap_size)

        per_token_data = []

        total_processed_tokens = 0  # Keep track of total tokens_logits processed to avoid duplicates

        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):

            with torch.no_grad():

                outputs = model(input_ids=chunk)
                logits = outputs.logits  # Shape: [1, seq_length, vocab_size]

            tokens = tokenizer.convert_ids_to_tokens(chunk.squeeze())
            num_tokens = len(tokens)

            chunk_data = []

            # Determine the starting index for predictions
            if i == 0:
                # For the first chunk, start from index 1 (since the first token has no previous context)
                start_idx = 1
            else:
                # For subsequent chunks, skip tokens_logits that were already processed in the overlap
                start_idx = overlap_size

            # Loop over the tokens_logits to predict
            for j in range(start_idx, num_tokens):
                # Compute per-token metrics
                per_token_metrics = self._compute_per_token_metrics(logits, chunk, tokens, j, tokenizer)
                chunk_data.append(per_token_metrics)

            # Append the chunk data to the per_token_data list
            per_token_data.extend(chunk_data)
            total_processed_tokens += len(chunk_data)

        return per_token_data

    def _compute_per_token_metrics(self, logits, chunk, tokens, j, tokenizer):

        # The model_instance predicts the token at position j using tokens_logits up to position j-1
        # Therefore, logits at position j-1 correspond to predictions for token at position j
        token_logits = logits[:, j - 1, :]  # Shape: [1, vocab_size]
        token_probs = F.softmax(token_logits, dim=-1)
        token_logprobs = F.log_softmax(token_logits, dim=-1)

        actual_token_id = chunk[:, j]  # The actual token at position j
        logprob_actual = token_logprobs[0, actual_token_id].item()
        max_logprob, max_token_id = torch.max(token_logprobs, dim=-1)
        max_logprob = max_logprob.item()
        max_token_id = max_token_id.item()
        entropy = -(token_probs * token_logprobs).sum().item()

        most_likely_token = tokenizer.convert_ids_to_tokens([max_token_id])[0]
        token = tokens[j]  # The token at position j

        return {
            'token': token,
            'logprob_actual': logprob_actual,
            'logprob_max': max_logprob,
            'entropy': entropy,
            'most_likely_token': most_likely_token
        }

    def _split_into_chunks(self, input_ids, chunk_size, overlap_size):

        input_ids = input_ids.squeeze()
        input_length = input_ids.size(0)
        stride = chunk_size - overlap_size
        chunks = []

        for i in range(0, input_length, stride):
            end_index = min(i + chunk_size, input_length)
            chunk = input_ids[i:end_index]
            chunks.append(chunk.unsqueeze(0).to(self.device))
            if end_index == input_length:
                break

        return chunks