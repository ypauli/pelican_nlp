from pelican_nlp.extraction.language_model import Model
from pelican_nlp.preprocessing.text_tokenizer import TextTokenizer
from pelican_nlp.preprocessing.text_cleaner import lowercase, remove_punctuation

from pelican_nlp.config import debug_print

class EmbeddingsExtractor:
    def __init__(self, embeddings_configurations, project_path):
        self.embeddings_configurations = embeddings_configurations
        self.model_name = embeddings_configurations['model_name']  # Embedding model instance (e.g., fastText, RoBERTa)
        self.model = Model(self.model_name, project_path)
        self.Tokenizer = TextTokenizer(self.embeddings_configurations['tokenization_method'], self.model_name,
                                       self.embeddings_configurations['max_length'])

        self.model.load_model()
        self.model_instance = self.model.model_instance

    def extract_embeddings_from_text(self, text_list, embedding_options):

        doc_entry_list = []

        for text in text_list:

            embeddings = {}

            # Tokenize the input text
            inputs = self.Tokenizer.tokenize_text(text)
            
            # Apply transformations based on embedding_options and tokenization method
            tokenization_method = self.embeddings_configurations['tokenization_method']
            
            if embedding_options.get('lowercase', False) or embedding_options.get('remove_punctuation', False):
                if tokenization_method == 'whitespace':
                    # For whitespace tokenization, inputs is a list of strings
                    # Apply transformations directly
                    if embedding_options.get('lowercase', False):
                        inputs = lowercase(inputs)
                    if embedding_options.get('remove_punctuation', False):
                        inputs = remove_punctuation(inputs)
                
                elif tokenization_method in ['model', 'model_roberta']:
                    # For model-based tokenization, need to convert token IDs to tokens,
                    # apply transformations, then re-tokenize
                    import torch
                    
                    # Extract token IDs based on input type
                    if tokenization_method == 'model_roberta':
                        # BatchEncoding object - extract token IDs
                        if hasattr(inputs, 'input_ids'):
                            # BatchEncoding has input_ids attribute
                            if hasattr(inputs['input_ids'], 'tolist'):
                                # It's a tensor, convert to list
                                token_ids = inputs['input_ids'][0].tolist() if len(inputs['input_ids'].shape) > 1 else inputs['input_ids'].tolist()
                            else:
                                # Already a list or array
                                token_ids = inputs['input_ids'][0] if isinstance(inputs['input_ids'], (list, tuple)) else inputs['input_ids']
                        elif isinstance(inputs, dict) and 'input_ids' in inputs:
                            # Dictionary with input_ids key
                            token_ids = inputs['input_ids'][0].tolist() if hasattr(inputs['input_ids'], 'tolist') else inputs['input_ids'][0]
                        else:
                            # Fallback: try to extract from first element if list/tensor
                            if hasattr(inputs, 'tolist'):
                                token_ids = inputs[0].tolist() if len(inputs.shape) > 1 else inputs.tolist()
                            elif isinstance(inputs, list):
                                token_ids = inputs[0] if inputs else []
                            else:
                                raise ValueError(f"Unable to extract token IDs from inputs of type {type(inputs)}")
                    else:
                        # 'model' method - inputs is a list of token IDs
                        token_ids = inputs
                    
                    # Convert token IDs to token strings
                    tokens = self.Tokenizer.tokenizer.convert_ids_to_tokens(token_ids)
                    
                    # Apply transformations to token strings
                    if embedding_options.get('lowercase', False):
                        tokens = lowercase(tokens)
                    if embedding_options.get('remove_punctuation', False):
                        tokens = remove_punctuation(tokens)
                    
                    # Remove empty tokens after punctuation removal
                    tokens = [token for token in tokens if token]
                    
                    # Re-tokenize the modified tokens
                    # Join tokens back to text, then re-tokenize with the model's tokenizer
                    # Note: This approach preserves subword tokenization where possible
                    if tokenization_method == 'model_roberta':
                        # For model_roberta, re-tokenize the modified token sequence
                        # We need to join and re-tokenize to get proper token IDs
                        reconstructed_text = self.Tokenizer.tokenizer.convert_tokens_to_string(tokens)
                        inputs = self.Tokenizer.tokenize_text(reconstructed_text)
                    else:
                        # For 'model' method, convert modified tokens back to token IDs
                        token_ids = self.Tokenizer.tokenizer.convert_tokens_to_ids(tokens)
                        inputs = token_ids

            debug_print(f'inputs are: {inputs}')
            debug_print(f'inputs type: {type(inputs)}')

            # Initialize embeddings list to ensure it's always defined
            embeddings = []
            
            if self.embeddings_configurations['pytorch_based_model']:
                #e.g. RoBERTa Model or Llama Model
                import torch
                try:
                    self.device = next(self.model_instance.parameters()).device
                except StopIteration:
                    self.device = torch.device("cpu")
                with torch.no_grad():
                    if 'llama' in self.model_name.lower():
                        # Handle Llama models which expect input_ids directly
                        outputs = self.model_instance(input_ids=inputs['input_ids'])
                    else:
                        # Handle RoBERTa and other models that accept **inputs
                        if isinstance(inputs, dict):
                            # Ensure inputs are on the same device as the model
                            inputs = {k: v.to(self.model_instance.device) for k, v in inputs.items()}
                            debug_print(f"Model inputs: {inputs}")
                            outputs = self.model_instance(**inputs, output_hidden_states=True)
                        else:
                            debug_print(f"Input type: {type(inputs)}")
                            debug_print(f"Input content: {inputs}")
                            
                            # Handle BatchEncoding objects from transformers
                            if hasattr(inputs, 'input_ids'):
                                # This is a BatchEncoding object, extract the tensors
                                input_ids = inputs['input_ids'].to(self.model_instance.device)
                                attention_mask = inputs['attention_mask'].to(self.model_instance.device) if 'attention_mask' in inputs else torch.ones_like(input_ids)
                                debug_print(f"Extracted from BatchEncoding - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
                                outputs = self.model_instance(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                            else:
                                # If inputs is a list of strings, convert to token IDs first
                                if isinstance(inputs, list):
                                    if isinstance(inputs[0], str):
                                        # Convert tokens to IDs
                                        token_ids = self.Tokenizer.tokenizer.convert_tokens_to_ids(inputs)
                                        debug_print(f"Token IDs: {token_ids}")
                                        inputs = torch.tensor([token_ids], device=self.model_instance.device)
                                    else:
                                        # If it's already a list of numbers, convert directly
                                        inputs = torch.tensor([inputs], device=self.model_instance.device)
                                else:
                                    # If it's already a tensor, just move to device
                                    inputs = inputs.to(self.model_instance.device)
                                
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
                    if hasattr(embedding, 'tolist'):
                        embedding_array = embedding.tolist()
                    elif hasattr(embedding, 'numpy'):
                        embedding_array = embedding.numpy().tolist()
                    elif isinstance(embedding, (list, tuple)):
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