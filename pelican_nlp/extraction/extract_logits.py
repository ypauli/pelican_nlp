import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

from pelican_nlp.config import debug_print
from pelican_nlp.extraction.token_artifacts import (
    DEFAULT_TRAILING_ARTIFACT_SEQUENCES,
    strip_trailing_artifact_items,
)
from pelican_nlp.extraction.sectioning import iter_section_groups
from pelican_nlp.extraction.resources import release_gpu
from pelican_nlp.preprocessing.text_tokenizer import TextTokenizer
from pelican_nlp.utils.csv_functions import store_features_to_csv

class LogitsExtractor:
    def __init__(self, options):

        self.options = options
        self.model_name = self.options['model_name']
        from pelican_nlp.utils.gpu_budget import apply_gpu_budget, runtime_torch_device
        apply_gpu_budget()
        self.device = runtime_torch_device(min_free_gb=1.0, allow_mps=False)
        self.trailing_artifact_token_sequences = self.options.get(
            'trailing_artifact_token_sequences',
            DEFAULT_TRAILING_ARTIFACT_SEQUENCES,
        )

    def extract_features(self, section, tokenizer, model):

        debug_print(f'section to tokenize: {section}')
        tokens = tokenizer.tokenize_text(section)
        debug_print(tokens)

        chunk_size = self.options['chunk_size']
        overlap_size = self.options['overlap_size']

        # Convert list of token IDs to tensor if needed
        if isinstance(tokens, list):
            input_ids = torch.tensor([tokens], device=self.device)
        elif hasattr(tokens, 'input_ids'):
            # Handle BatchEncoding objects (tokenization_method: model)
            input_ids = tokens['input_ids'].to(self.device)
        elif isinstance(tokens, dict) and 'input_ids' in tokens:
            # Handle dictionary with input_ids key
            input_ids = tokens['input_ids'].to(self.device)
        else:
            input_ids = tokens.to(self.device)
            
        chunks = self._split_into_chunks(input_ids, chunk_size, overlap_size)

        per_token_data = []

        total_processed_tokens = 0  # Keep track of total tokens_logits processed to avoid duplicates

        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks", file=sys.stderr, mininterval=1.0)):

            with torch.inference_mode():

                outputs = model(input_ids=chunk)
                logits = outputs.logits  # Shape: [1, seq_length, vocab_size]

            # Safely convert chunk to list of token IDs, ensuring we don't create 0-d tensor
            # Remove batch dimension if present (chunk should be [1, seq_len] from _split_into_chunks)
            if chunk.dim() == 2 and chunk.size(0) == 1:
                # Remove batch dimension: [1, seq_len] -> [seq_len]
                chunk_ids = chunk[0]  # Use indexing instead of squeeze to avoid 0-d issues
            elif chunk.dim() == 1:
                chunk_ids = chunk
            else:
                # Fallback: ensure it's at least 1D
                chunk_ids = chunk.view(-1) if chunk.numel() > 0 else torch.tensor([], dtype=chunk.dtype, device=chunk.device)
            
            # Convert to list - tolist() on 1D tensor always returns a list
            chunk_ids_list = chunk_ids.tolist()
            # Double-check: ensure it's always a list (handle any edge cases)
            if not isinstance(chunk_ids_list, list):
                chunk_ids_list = [chunk_ids_list]
            tokens = tokenizer.convert_ids_to_tokens(chunk_ids_list)
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

        per_token_data = self._remove_trailing_artifact_tokens(per_token_data)
        return per_token_data

    def process_corpus(self, corpus):
        """Load a causal LM and write logits for every document section."""
        from pelican_nlp.extraction.language_model import Model
        from pelican_nlp.extraction.model_registry import MODEL_KIND_CAUSAL_LM

        model_name = self.options["model_name"]
        trust_remote_code = self.options.get("trust_remote_code", False)
        model = Model(
            model_name,
            model_kind=self.options.get("model_kind"),
        )
        model.load_model(trust_remote_code=trust_remote_code)
        from pelican_nlp.utils.gpu_budget import input_device_for_model
        self.device = input_device_for_model(model.model_instance)
        if model.kind != MODEL_KIND_CAUSAL_LM:
            raise ValueError(
                f"Logits extraction requires a causal language model, but '{model_name}' "
                f"loaded as '{model.kind}'. Use a decoder model (e.g. Llama) in "
                "options_logits.model_name, or set options_logits.model_kind: causal_lm "
                "only for decoder checkpoints."
            )
        tokenizer = TextTokenizer(
            self.options["tokenization_method"],
            model_name=self.options["model_name"],
            trust_remote_code=trust_remote_code,
        )
        print(
            f"Extracting logits with {model_name} on {self.device}. "
            f"{len(corpus.documents)} document(s).",
            flush=True,
        )
        keep_speakertags = self.options.get("keep_speakertags", False)
        try:
            total = len(corpus.documents)
            for index, document in enumerate(corpus.documents, start=1):
                print(
                    f"Logits [{index}/{total}] {document.name}",
                    flush=True,
                )
                for key, section_parts in iter_section_groups(
                    document, corpus.config, keep_speakertags=keep_speakertags
                ):
                    print(
                        f"  section {key}: {len(section_parts)} part(s)",
                        flush=True,
                    )
                    for part in section_parts:
                        logits = self.extract_features(part, tokenizer, model.model_instance)
                        document.logits.append(logits)
                        store_features_to_csv(
                            logits,
                            corpus.derivatives_dir,
                            document,
                            metric="logits",
                        )
        finally:
            release_gpu(model, tokenizer, self)
            print("GPU memory cleared after logits extraction", flush=True)

    def _remove_trailing_artifact_tokens(self, per_token_data):
        return strip_trailing_artifact_items(
            per_token_data,
            sequences=self.trailing_artifact_token_sequences,
            token_of=lambda item: item.get("token", ""),
        )

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

        # Squeeze batch dimension if present, but ensure we keep at least 1D
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze()
        # Ensure input_ids is at least 1D (handle edge case where squeeze removed all dimensions)
        if input_ids.dim() == 0:
            input_ids = input_ids.unsqueeze(0)
        
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