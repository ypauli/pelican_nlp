import pandas as pd

from pelican_nlp.extraction.model_registry import MODEL_KIND_CAUSAL_LM, MODEL_KIND_STATIC
from pelican_nlp.preprocessing.text_tokenizer import TextTokenizer
from pelican_nlp.preprocessing.text_cleaner import lowercase, remove_punctuation
from pelican_nlp.preprocessing import text_cleaner as textcleaner
from pelican_nlp.utils.csv_functions import store_features_to_csv
from pelican_nlp.extraction.sectioning import iter_section_groups
from pelican_nlp.extraction.resources import release_gpu

from pelican_nlp.config import debug_print


def _embedding_batch_size(embedding_options) -> int:
    try:
        value = int(embedding_options.get("batch_size", 1) or 1)
    except (TypeError, ValueError):
        return 1
    return max(1, value)


class EmbeddingsExtractor:
    def __init__(self, embeddings_configurations):
        self.embeddings_configurations = embeddings_configurations
        self.model_name = embeddings_configurations['model_name']  # Embedding model instance (e.g., fastText, RoBERTa)
        trust_remote_code = embeddings_configurations.get('trust_remote_code', False)
        from pelican_nlp.extraction.language_model import Model
        self.model = Model(
            self.model_name,
            model_kind=embeddings_configurations.get('model_kind'),
        )
        self.Tokenizer = TextTokenizer(
            self.embeddings_configurations['tokenization_method'],
            self.model_name,
            self.embeddings_configurations['max_length'],
            trust_remote_code=trust_remote_code,
        )

        self.model.load_model(trust_remote_code=trust_remote_code)
        self.model_instance = self.model.model_instance
        if 'pytorch_based_model' not in self.embeddings_configurations:
            self.embeddings_configurations['pytorch_based_model'] = self.model.kind != MODEL_KIND_STATIC

    def extract_embeddings_from_text(self, text_list, embedding_options):

        batch_size = _embedding_batch_size(embedding_options)
        if (
            self.embeddings_configurations['pytorch_based_model']
            and batch_size > 1
            and self.model.kind != MODEL_KIND_CAUSAL_LM
            and len(text_list) > 1
        ):
            return self._extract_embeddings_encoder_batched(
                text_list, embedding_options, batch_size
            )

        doc_entry_list = []

        for text in text_list:

            embeddings = {}

            inputs = self._prepare_embedding_inputs(text, embedding_options)

            debug_print(f'inputs are: {inputs}')
            debug_print(f'inputs type: {type(inputs)}')

            # Initialize embeddings list to ensure it's always defined
            embeddings = []
            
            if self.embeddings_configurations['pytorch_based_model']:
                #e.g. RoBERTa Model or Llama Model
                import torch
                from pelican_nlp.utils.gpu_budget import input_device_for_model
                try:
                    self.device = input_device_for_model(self.model_instance)
                except (StopIteration, TypeError):
                    self.device = torch.device("cpu")
                with torch.inference_mode():
                    if self.model.kind == MODEL_KIND_CAUSAL_LM:
                        # Causal LMs (Llama and others) which expect input_ids directly
                        outputs = self.model_instance(input_ids=inputs['input_ids'])
                    else:
                        # Handle RoBERTa and other models that accept **inputs
                        if isinstance(inputs, dict):
                            # Ensure inputs are on the same device as the model
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}
                            debug_print(f"Model inputs: {inputs}")
                            outputs = self.model_instance(**inputs, output_hidden_states=True)
                        else:
                            debug_print(f"Input type: {type(inputs)}")
                            debug_print(f"Input content: {inputs}")
                            
                            # Handle BatchEncoding objects from transformers
                            if hasattr(inputs, 'input_ids'):
                                # This is a BatchEncoding object, extract the tensors
                                input_ids = inputs['input_ids'].to(self.device)
                                attention_mask = inputs['attention_mask'].to(self.device) if 'attention_mask' in inputs else torch.ones_like(input_ids)
                                debug_print(f"Extracted from BatchEncoding - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
                                outputs = self.model_instance(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                            else:
                                # If inputs is a list of strings, convert to token IDs first
                                if isinstance(inputs, list):
                                    if isinstance(inputs[0], str):
                                        # Convert tokens to IDs
                                        token_ids = self.Tokenizer.tokenizer.convert_tokens_to_ids(inputs)
                                        debug_print(f"Token IDs: {token_ids}")
                                        inputs = torch.tensor([token_ids], device=self.device)
                                    else:
                                        # If it's already a list of numbers, convert directly
                                        inputs = torch.tensor([inputs], device=self.device)
                                else:
                                    # If it's already a tensor, just move to device
                                    inputs = inputs.to(self.device)
                                
                                # Only print shape if inputs is a tensor
                                if hasattr(inputs, 'shape'):
                                    debug_print(f"Final tensor shape: {inputs.shape}")
                                else:
                                    debug_print(f"Inputs is not a tensor, type: {type(inputs)}")
                                
                                # Ensure proper shape
                                if len(inputs.shape) == 1:
                                    inputs = inputs.unsqueeze(0)  # Add batch dimension
                                
                                # Create attention mask
                                attention_mask = torch.ones_like(inputs)
                                debug_print(f"Model inputs - input_ids: {inputs.shape}, attention_mask: {attention_mask.shape}")
                                outputs = self.model_instance(input_ids=inputs, attention_mask=attention_mask, output_hidden_states=True)
                                debug_print(f"Model outputs type: {type(outputs)}")
                                debug_print(f"Model outputs attributes: {dir(outputs)}")

                # Get word embeddings (last hidden state)
                if outputs is None:
                    raise ValueError("Model returned None output")
                
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    word_embeddings = outputs.hidden_states[-1]
                    debug_print(f"Using hidden_states, shape: {word_embeddings.shape}")
                elif hasattr(outputs, 'last_hidden_state'):
                    word_embeddings = outputs.last_hidden_state
                    debug_print(f"Using last_hidden_state, shape: {word_embeddings.shape}")
                else:
                    raise ValueError(f"Model output has neither hidden_states nor last_hidden_state. Available attributes: {dir(outputs)}")
                
                # Don't modify word_embeddings here - it's a view of the model output
                # Moving it to CPU at this point can cause segfaults if the model output is still referenced
                # Instead, we'll convert individual embeddings to CPU when processing them

                # Extract input_ids and attention_mask to identify padding tokens
                if isinstance(inputs, dict):
                    input_ids = inputs['input_ids'][0].tolist()
                    attention_mask = inputs.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask[0].tolist() if hasattr(attention_mask, 'tolist') else attention_mask
                elif hasattr(inputs, 'input_ids'):
                    # Handle BatchEncoding objects (they are dict-like)
                    input_ids = inputs['input_ids'][0].tolist()
                    # BatchEncoding objects support dict-like access
                    if 'attention_mask' in inputs:
                        attention_mask = inputs['attention_mask']
                        attention_mask = attention_mask[0].tolist() if hasattr(attention_mask, 'tolist') else attention_mask
                    elif hasattr(inputs, 'attention_mask'):
                        attention_mask = inputs.attention_mask
                        attention_mask = attention_mask[0].tolist() if hasattr(attention_mask, 'tolist') else attention_mask
                    else:
                        attention_mask = None
                else:
                    input_ids = inputs[0].tolist()
                    attention_mask = None
                
                tokens = self.Tokenizer.tokenizer.convert_ids_to_tokens(input_ids)
                
                # Get special token IDs to identify tokens that should be filtered
                pad_token_id = self.Tokenizer.tokenizer.pad_token_id if hasattr(self.Tokenizer.tokenizer, 'pad_token_id') else None
                pad_token = self.Tokenizer.tokenizer.pad_token if hasattr(self.Tokenizer.tokenizer, 'pad_token') else None
                unk_token_id = self.Tokenizer.tokenizer.unk_token_id if hasattr(self.Tokenizer.tokenizer, 'unk_token_id') else None
                unk_token = self.Tokenizer.tokenizer.unk_token if hasattr(self.Tokenizer.tokenizer, 'unk_token') else None
                
                debug_print(f"[extract_embeddings] Total tokens before filtering: {len(tokens)}")
                debug_print(f"[extract_embeddings] Pad token ID: {pad_token_id}, Pad token: {pad_token}")
                debug_print(f"[extract_embeddings] UNK token ID: {unk_token_id}, UNK token: {unk_token}")
                if attention_mask is not None:
                    padding_count = sum(1 for m in attention_mask if m == 0)
                    debug_print(f"[extract_embeddings] Attention mask padding count: {padding_count}")
                
                # Now align the tokens and embeddings, filtering out padding tokens and zero vectors
                embeddings = []
                padding_filtered = 0
                zero_vector_filtered = 0
                special_token_filtered = 0
                
                for idx, (token, embedding) in enumerate(zip(tokens, word_embeddings[0])):
                    # Check if this is a padding token
                    is_padding = False
                    is_special = False
                    
                    # Method 1: Check attention mask (most reliable)
                    if attention_mask is not None and idx < len(attention_mask):
                        if attention_mask[idx] == 0:
                            is_padding = True
                            debug_print(f"[extract_embeddings] Token {idx} '{token}' is padding (attention_mask=0)")
                    
                    # Method 2: Check if token ID matches pad_token_id
                    if not is_padding and pad_token_id is not None and idx < len(input_ids):
                        if input_ids[idx] == pad_token_id:
                            is_padding = True
                            debug_print(f"[extract_embeddings] Token {idx} '{token}' is padding (pad_token_id match)")
                    
                    # Method 3: Check if token string matches pad_token
                    if not is_padding and pad_token is not None:
                        if token == pad_token:
                            is_padding = True
                            debug_print(f"[extract_embeddings] Token {idx} '{token}' is padding (pad_token match)")
                    
                    # Check for unknown tokens
                    if not is_padding and unk_token_id is not None and idx < len(input_ids):
                        if input_ids[idx] == unk_token_id:
                            is_special = True
                            debug_print(f"[extract_embeddings] Token {idx} '{token}' is UNK token (unk_token_id match)")
                    
                    if not is_padding and unk_token is not None:
                        if token == unk_token:
                            is_special = True
                            debug_print(f"[extract_embeddings] Token {idx} '{token}' is UNK token (unk_token match)")
                    
                    # Check for zero vectors - convert to array first for robust checking
                    # Convert tensor to list/array, moving to CPU if needed
                    # This creates a copy, avoiding issues with modifying views
                    try:
                        if hasattr(embedding, 'tolist'):
                            # tolist() automatically moves to CPU if needed
                            embedding_array = embedding.tolist()
                        elif hasattr(embedding, 'cpu'):
                            # If it's a tensor, move to CPU and convert
                            embedding_array = embedding.cpu().detach().clone().tolist()
                        elif hasattr(embedding, 'numpy'):
                            embedding_array = embedding.numpy().tolist()
                        elif isinstance(embedding, (list, tuple)):
                            embedding_array = list(embedding)
                        else:
                            embedding_array = embedding
                    except Exception as e:
                        # Fallback: try to convert directly
                        debug_print(f"Warning: Could not convert embedding to list: {e}")
                        if isinstance(embedding, (list, tuple)):
                            embedding_array = list(embedding)
                        else:
                            embedding_array = embedding
                    
                    # More robust zero vector detection using numpy or manual calculation
                    try:
                        import numpy as np
                        if isinstance(embedding_array, list):
                            embedding_np = np.array(embedding_array)
                        else:
                            embedding_np = np.array(embedding_array)
                        embedding_norm = np.linalg.norm(embedding_np)
                        # Use a small epsilon to account for floating point precision
                        is_zero_vector = embedding_norm < 1e-10
                    except:
                        # Fallback to manual calculation
                        if isinstance(embedding_array, list):
                            embedding_norm = sum(x*x for x in embedding_array)
                        else:
                            embedding_norm = 0.0
                        is_zero_vector = embedding_norm < 1e-20
                    
                    # Skip padding tokens
                    if is_padding:
                        padding_filtered += 1
                        debug_print(f"[extract_embeddings] Filtering out padding token at index {idx}: '{token}'")
                        continue
                    
                    # Filter out zero vectors (they cause issues in similarity calculations)
                    if is_zero_vector:
                        zero_vector_filtered += 1
                        token_id = input_ids[idx] if idx < len(input_ids) else None
                        debug_print(f"[extract_embeddings] ERROR: Zero vector detected at index {idx} for token '{token}' (ID: {token_id})")
                        debug_print(f"[extract_embeddings] Filtering out zero vector token '{token}' - this will cause NaN in window similarity calculations")
                        continue
                    
                    # Optionally filter special tokens (but keep them if they have non-zero embeddings)
                    if is_special:
                        special_token_filtered += 1
                        debug_print(f"[extract_embeddings] Note: Special token '{token}' at index {idx} has non-zero embedding, keeping it")
                    
                    embeddings.append((token, embedding_array))
                
                debug_print(f"[extract_embeddings] Filtered {padding_filtered} padding tokens")
                debug_print(f"[extract_embeddings] Filtered {zero_vector_filtered} zero vector tokens")
                debug_print(f"[extract_embeddings] Found {special_token_filtered} special tokens (kept if non-zero)")
                debug_print(f"[extract_embeddings] Final embeddings count: {len(embeddings)}")
                
                if padding_filtered > 0:
                    debug_print(f"[extract_embeddings] IMPORTANT: Removed {padding_filtered} padding token(s)")
                if zero_vector_filtered > 0:
                    debug_print(f"[extract_embeddings] CRITICAL: Removed {zero_vector_filtered} zero vector token(s) that would cause NaN in window similarity calculations")

            else:
                if self.model_name == 'fastText':
                    # Clear embeddings list for fastText (it was initialized above)
                    embeddings = []
                    zero_vector_filtered = 0
                    for token in inputs:
                        embedding = self.model_instance.get_word_vector(token)
                        # Check for zero vectors in fastText embeddings too
                        try:
                            import numpy as np
                            embedding_np = np.array(embedding)
                            embedding_norm = np.linalg.norm(embedding_np)
                            is_zero_vector = embedding_norm < 1e-10
                        except:
                            # Fallback check
                            if isinstance(embedding, (list, tuple)):
                                embedding_norm = sum(x*x for x in embedding)
                            else:
                                embedding_norm = 0.0
                            is_zero_vector = embedding_norm < 1e-20
                        
                        if is_zero_vector:
                            zero_vector_filtered += 1
                            debug_print(f"[extract_embeddings] ERROR: Zero vector detected for fastText token '{token}'")
                            debug_print(f"[extract_embeddings] Filtering out zero vector token '{token}'")
                            continue
                        
                        embeddings.append((token, embedding))
                    
                    if zero_vector_filtered > 0:
                        debug_print(f"[extract_embeddings] CRITICAL: Removed {zero_vector_filtered} zero vector token(s) from fastText embeddings")

            # Final pass: Remove any remaining zero vectors that might have slipped through
            final_embeddings = []
            final_zero_filtered = 0
            for token, embedding in embeddings:
                try:
                    import numpy as np
                    if isinstance(embedding, list):
                        embedding_np = np.array(embedding)
                    else:
                        embedding_np = np.array(embedding)
                    embedding_norm = np.linalg.norm(embedding_np)
                    is_zero_vector = embedding_norm < 1e-10
                except:
                    # Fallback check
                    if isinstance(embedding, (list, tuple)):
                        embedding_norm = sum(x*x for x in embedding)
                    else:
                        embedding_norm = 0.0
                    is_zero_vector = embedding_norm < 1e-20
                
                if is_zero_vector:
                    final_zero_filtered += 1
                    debug_print(f"[extract_embeddings] FINAL PASS: Removing zero vector for token '{token}'")
                    continue
                
                final_embeddings.append((token, embedding))
            
            if final_zero_filtered > 0:
                debug_print(f"[extract_embeddings] FINAL PASS: Removed {final_zero_filtered} additional zero vector(s)")
                debug_print(f"[extract_embeddings] Final embeddings after all filtering: {len(final_embeddings)}")
            
            doc_entry_list.append(final_embeddings)

        # Calculate token count properly
        if isinstance(inputs, dict):
            token_count = len(inputs['input_ids'][0]) if 'input_ids' in inputs else 0
        elif hasattr(inputs, 'input_ids'):
            # Handle BatchEncoding objects
            token_count = len(inputs['input_ids'][0]) if hasattr(inputs['input_ids'], '__len__') else 0
        elif hasattr(inputs, '__len__'):
            token_count = len(inputs)
        else:
            token_count = 0

        return doc_entry_list, token_count

    def _extract_embeddings_encoder_batched(self, text_list, embedding_options, batch_size):
        """One encoder forward per chunk of texts. Opt-in via options_embeddings.batch_size."""
        import torch
        from types import SimpleNamespace

        prepared = [
            self._prepare_embedding_inputs(text, embedding_options) for text in text_list
        ]
        doc_entry_list = []
        with torch.inference_mode():
            for start in range(0, len(prepared), batch_size):
                chunk = prepared[start:start + batch_size]
                hidden_rows = self._encoder_last_hidden_batched(chunk)
                for inputs, hidden in zip(chunk, hidden_rows):
                    outputs = SimpleNamespace(
                        hidden_states=(hidden,),
                        last_hidden_state=hidden,
                    )
                    doc_entry_list.append(
                        self._collect_pytorch_token_embeddings(inputs, outputs)
                    )
        inputs = prepared[-1] if prepared else None
        if isinstance(inputs, dict):
            token_count = len(inputs['input_ids'][0]) if 'input_ids' in inputs else 0
        elif hasattr(inputs, 'input_ids'):
            token_count = len(inputs['input_ids'][0]) if hasattr(inputs['input_ids'], '__len__') else 0
        elif inputs is not None and hasattr(inputs, '__len__'):
            token_count = len(inputs)
        else:
            token_count = 0
        return doc_entry_list, token_count

    def _prepare_embedding_inputs(self, text, embedding_options):
        from pelican_nlp.preprocessing.text_tokenizer import TOKENIZATION_WHITESPACE

        inputs = self.Tokenizer.tokenize_text(text)
        if not (
            embedding_options.get("lowercase", False)
            or embedding_options.get("remove_punctuation", False)
        ):
            return inputs

        if self.Tokenizer.tokenization_method == TOKENIZATION_WHITESPACE:
            if embedding_options.get("lowercase", False):
                inputs = lowercase(inputs)
            if embedding_options.get("remove_punctuation", False):
                inputs = remove_punctuation(inputs)
            return inputs

        token_ids = inputs["input_ids"]
        if hasattr(token_ids, "tolist"):
            token_ids = (
                token_ids[0].tolist() if len(token_ids.shape) > 1 else token_ids.tolist()
            )
        elif isinstance(token_ids, (list, tuple)):
            token_ids = (
                token_ids[0]
                if token_ids and isinstance(token_ids[0], (list, tuple))
                else token_ids
            )
        tokens = self.Tokenizer.tokenizer.convert_ids_to_tokens(token_ids)
        if embedding_options.get("lowercase", False):
            tokens = lowercase(tokens)
        if embedding_options.get("remove_punctuation", False):
            tokens = remove_punctuation(tokens)
        tokens = [token for token in tokens if token]
        reconstructed_text = self.Tokenizer.tokenizer.convert_tokens_to_string(tokens)
        return self.Tokenizer.tokenize_text(reconstructed_text)

    def _encoder_last_hidden_batched(self, inputs_list):
        import torch
        from pelican_nlp.utils.gpu_budget import input_device_for_model

        device = input_device_for_model(self.model_instance)
        pad_id = getattr(self.Tokenizer.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        id_rows = []
        mask_rows = []
        lengths = []
        for inputs in inputs_list:
            if hasattr(inputs, "input_ids") or (isinstance(inputs, dict) and "input_ids" in inputs):
                ids = inputs["input_ids"]
            else:
                ids = inputs
            if hasattr(ids, "dim"):
                row = ids.to(device)
                if row.dim() > 1:
                    row = row[0]
            else:
                values = ids[0] if ids and isinstance(ids[0], (list, tuple)) else ids
                row = torch.tensor(values, device=device)
            row = row.view(-1)
            lengths.append(int(row.size(0)))
            id_rows.append(row)
            if hasattr(inputs, "attention_mask") or (isinstance(inputs, dict) and "attention_mask" in inputs):
                mask = inputs["attention_mask"]
                if hasattr(mask, "to"):
                    mask_row = mask.to(device)
                    if mask_row.dim() > 1:
                        mask_row = mask_row[0]
                else:
                    mask_row = torch.tensor(mask, device=device)
                mask_rows.append(mask_row.view(-1))
            else:
                mask_rows.append(torch.ones_like(row))
        padded_ids = torch.nn.utils.rnn.pad_sequence(id_rows, batch_first=True, padding_value=pad_id)
        padded_mask = torch.nn.utils.rnn.pad_sequence(mask_rows, batch_first=True, padding_value=0)
        outputs = self.model_instance(
            input_ids=padded_ids,
            attention_mask=padded_mask,
            output_hidden_states=True,
        )
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        else:
            raise ValueError("Batched encoder output has neither hidden_states nor last_hidden_state.")
        return [hidden[i : i + 1, :length, :] for i, length in enumerate(lengths)]

    def _collect_pytorch_token_embeddings(self, inputs, outputs):
        if outputs is None:
            raise ValueError("Model returned None output")
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            word_embeddings = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            word_embeddings = outputs.last_hidden_state
        else:
            raise ValueError(
                f"Model output has neither hidden_states nor last_hidden_state. Available attributes: {dir(outputs)}"
            )
        if isinstance(inputs, dict):
            input_ids = inputs['input_ids'][0].tolist()
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask[0].tolist() if hasattr(attention_mask, 'tolist') else attention_mask
        elif hasattr(inputs, 'input_ids'):
            input_ids = inputs['input_ids'][0].tolist()
            if 'attention_mask' in inputs:
                attention_mask = inputs['attention_mask']
                attention_mask = attention_mask[0].tolist() if hasattr(attention_mask, 'tolist') else attention_mask
            elif hasattr(inputs, 'attention_mask'):
                attention_mask = inputs.attention_mask
                attention_mask = attention_mask[0].tolist() if hasattr(attention_mask, 'tolist') else attention_mask
            else:
                attention_mask = None
        else:
            input_ids = inputs[0].tolist()
            attention_mask = None
        tokens = self.Tokenizer.tokenizer.convert_ids_to_tokens(input_ids)
        pad_token_id = self.Tokenizer.tokenizer.pad_token_id if hasattr(self.Tokenizer.tokenizer, 'pad_token_id') else None
        pad_token = self.Tokenizer.tokenizer.pad_token if hasattr(self.Tokenizer.tokenizer, 'pad_token') else None
        unk_token_id = self.Tokenizer.tokenizer.unk_token_id if hasattr(self.Tokenizer.tokenizer, 'unk_token_id') else None
        unk_token = self.Tokenizer.tokenizer.unk_token if hasattr(self.Tokenizer.tokenizer, 'unk_token') else None
        embeddings = []
        for idx, (token, embedding) in enumerate(zip(tokens, word_embeddings[0])):
            is_padding = False
            if attention_mask is not None and idx < len(attention_mask) and attention_mask[idx] == 0:
                is_padding = True
            if not is_padding and pad_token_id is not None and idx < len(input_ids) and input_ids[idx] == pad_token_id:
                is_padding = True
            if not is_padding and pad_token is not None and token == pad_token:
                is_padding = True
            try:
                if hasattr(embedding, 'tolist'):
                    embedding_array = embedding.tolist()
                elif hasattr(embedding, 'cpu'):
                    embedding_array = embedding.cpu().detach().clone().tolist()
                else:
                    embedding_array = list(embedding) if isinstance(embedding, (list, tuple)) else embedding
            except Exception:
                embedding_array = embedding
            try:
                import numpy as np
                embedding_norm = float(np.linalg.norm(np.array(embedding_array)))
                is_zero_vector = embedding_norm < 1e-10
            except Exception:
                embedding_norm = sum(x * x for x in embedding_array) if isinstance(embedding_array, list) else 0.0
                is_zero_vector = embedding_norm < 1e-20
            if is_padding or is_zero_vector:
                continue
            embeddings.append((token, embedding_array))
        final_embeddings = []
        for token, embedding in embeddings:
            try:
                import numpy as np
                is_zero_vector = float(np.linalg.norm(np.array(embedding))) < 1e-10
            except Exception:
                is_zero_vector = False
            if not is_zero_vector:
                final_embeddings.append((token, embedding))
        return final_embeddings

    def process_corpus(self, corpus):
        """Extract embeddings for every document section and write derived metrics."""
        embedding_options = corpus.config["options_embeddings"]
        semantic_similarity_options = corpus.config.get("options_semantic-similarity", {})
        store_window_details = semantic_similarity_options.get("store_window_details", False)
        store_sentence_details = semantic_similarity_options.get(
            "store_sentence_details", store_window_details
        )
        keep_speakertags = embedding_options.get("keep_speakertags", False)
        run_distance = embedding_options.get("distance-from-randomness", False)

        debug_print(len(corpus.documents))
        total = len(corpus.documents)
        for index, document in enumerate(corpus.documents, start=1):
            print(f"Embeddings [{index}/{total}] {document.name}", flush=True)
            debug_print(f"cleaned sections: {document.cleaned_sections}")
            for key, section_parts in iter_section_groups(
                document, corpus.config, keep_speakertags=keep_speakertags
            ):
                debug_print(f"Processing section {key}")
                embeddings, token_count = self.extract_embeddings_from_text(
                    section_parts, embedding_options
                )
                document.embeddings.append(embeddings)

                if corpus.task == "fluency":
                    document.fluency_word_count = token_count

                for utterance_idx, utterance in enumerate(embeddings):
                    source_text_for_sentence_similarity = (
                        section_parts[utterance_idx] if utterance_idx < len(section_parts) else None
                    )
                    if embedding_options.get("semantic-similarity"):
                        _store_semantic_similarity(
                            utterance,
                            document,
                            corpus,
                            semantic_similarity_options,
                            store_window_details,
                            store_sentence_details,
                            source_text_for_sentence_similarity,
                        )

                    if run_distance:
                        from pelican_nlp.extraction.distance_from_randomness import (
                            get_distance_from_randomness,
                        )

                        divergence = get_distance_from_randomness(
                            utterance, corpus.config["options_dis_from_randomness"]
                        )
                        debug_print(f"Divergence from optimality metrics: {divergence}")
                        store_features_to_csv(
                            divergence,
                            corpus.derivatives_dir,
                            document,
                            metric="distance-from-randomness",
                        )

                    cleaned_embeddings = _clean_embedding_tokens(
                        utterance, embedding_options, corpus.config
                    )
                    store_features_to_csv(
                        cleaned_embeddings,
                        corpus.derivatives_dir,
                        document,
                        metric="embeddings",
                    )

        release_gpu(self)
        print("GPU memory cleared after embeddings extraction")
        return


def _store_semantic_similarity(
    utterance,
    document,
    corpus,
    semantic_similarity_options,
    store_window_details,
    store_sentence_details,
    source_text_for_sentence_similarity,
):
    from pelican_nlp.extraction.semantic_similarity import (
        calculate_semantic_similarity,
        get_semantic_similarity_windows,
        filter_punctuation_tokens,
    )

    exclude_punctuation_tokens = semantic_similarity_options.get(
        "exclude_punctuation_tokens", False
    )
    similarity_utterance = (
        filter_punctuation_tokens(utterance) if exclude_punctuation_tokens else utterance
    )
    consecutive_similarities, mean_similarity = calculate_semantic_similarity(similarity_utterance)
    debug_print(f"Mean semantic similarity: {mean_similarity:.4f}")

    for window_size in corpus.config["options_semantic-similarity"]["window_sizes"]:
        if window_size == "sentence":
            continue
        debug_print(
            f"\n[extract_embeddings] Processing window_size={window_size} for document: {document.name}"
        )
        collect_details = store_window_details and window_size != "sentence"
        window_input = similarity_utterance if window_size != "sentence" else utterance
        window_result = get_semantic_similarity_windows(
            window_input,
            window_size,
            return_details=collect_details,
            exclude_punctuation=exclude_punctuation_tokens,
        )

        if collect_details:
            window_stats, window_detail_rows = window_result
        else:
            window_stats = window_result
            window_detail_rows = []

        if isinstance(window_stats, tuple) and len(window_stats) == 5:
            window_data = {
                "mean_of_window_means": window_stats[0],
                "std_of_window_means": window_stats[1],
                "mean_of_window_stds": window_stats[2],
                "std_of_window_stds": window_stats[3],
                "mean_of_window_medians": window_stats[4],
            }
            nan_metrics = [k for k, v in window_data.items() if pd.isna(v)]
            if nan_metrics:
                debug_print(
                    f"[extract_embeddings] WARNING: Window {window_size} has NaN values for metrics: {nan_metrics}"
                )
        else:
            window_data = {
                "mean": window_stats[0] if isinstance(window_stats, tuple) else window_stats,
                "std": window_stats[1]
                if isinstance(window_stats, tuple) and len(window_stats) > 1
                else None,
            }

        store_features_to_csv(
            window_data,
            corpus.derivatives_dir,
            document,
            metric=f"semantic-similarity-window-{window_size}",
        )

        if collect_details and window_detail_rows:
            store_features_to_csv(
                window_detail_rows,
                corpus.derivatives_dir,
                document,
                metric=f"semantic-similarity-window-details-{window_size}",
            )

    if "sentence" in corpus.config["options_semantic-similarity"]["window_sizes"]:
        sentence_result = get_semantic_similarity_windows(
            utterance,
            "sentence",
            return_details=store_sentence_details,
            exclude_punctuation=exclude_punctuation_tokens,
            source_text=source_text_for_sentence_similarity,
        )
        if store_sentence_details:
            sentence_stats, sentence_detail_rows = sentence_result
        else:
            sentence_stats = sentence_result
            sentence_detail_rows = []
        if isinstance(sentence_stats, tuple) and len(sentence_stats) == 5:
            sentence_data = {
                "mean_of_window_means": sentence_stats[0],
                "std_of_window_means": sentence_stats[1],
                "mean_of_window_stds": sentence_stats[2],
                "std_of_window_stds": sentence_stats[3],
                "mean_of_window_medians": sentence_stats[4],
            }
            store_features_to_csv(
                sentence_data,
                corpus.derivatives_dir,
                document,
                metric="semantic-similarity-sentence",
            )
            if store_sentence_details and sentence_detail_rows:
                store_features_to_csv(
                    sentence_detail_rows,
                    corpus.derivatives_dir,
                    document,
                    metric="semantic-similarity-sentence-details",
                )


def _clean_embedding_tokens(utterance, embedding_options, config):
    if not embedding_options.get("clean_embedding_tokens"):
        return utterance if isinstance(utterance, list) else [(k, v) for k, v in utterance.items()]

    cleaned_embeddings = []
    model_name = str(config.get("options_embeddings", {}).get("model_name", "")).lower()
    if isinstance(utterance, dict):
        for token, embedding in utterance.items():
            if "xlm-roberta-base" in model_name:
                cleaned_token = textcleaner.clean_subword_token_RoBERTa(token)
            else:
                cleaned_token = textcleaner.clean_token_generic(token)
            if cleaned_token is not None:
                cleaned_embeddings.append((cleaned_token, embedding))
    else:
        for token, embedding in utterance:
            cleaned_token = textcleaner.clean_token_generic(token)
            if cleaned_token is not None:
                cleaned_embeddings.append((cleaned_token, embedding))
    return cleaned_embeddings