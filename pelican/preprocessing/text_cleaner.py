import string
import re
from collections import Counter

class TextCleaner:
    def __init__(self, options=None):
        self.options = options

    def clean(self, document, text, characters_to_remove=None):
        """Clean text based on configured options.
        
        Args:
            document: Document object containing metadata
            text: Text to clean
            characters_to_remove: Optional string of characters to remove
        Returns:
            Cleaned text string
        """
        if self.options.get('remove_timestamps', True):
            text = remove_timestamps(text, self.options.get('timestamp_pattern_example'))

        if characters_to_remove:  # Simplified condition
            text = _remove_special_characters(text, characters_to_remove)

        # Consolidate conditional blocks for better readability
        cleaning_operations = [
            ('fluency_task', lambda: self.clean_fluency_transcripts(document, text)),
            ('remove_punctuation', lambda: remove_punctuation(text)),
            ('lowercase', lambda: lowercase(text))
        ]

        for option, operation in cleaning_operations:
            if self.options.get(option):
                text = operation()

        if self.options.get('general_cleaning', True):
            replacements = [
                (r'/', ''),
                (r'\s+([?.!,"])', r'\1'),
                (r'\n\s*\n', '\n'),
                (r'\\', ''),
                (' +', ' ')
            ]
            text = self._apply_replacements(text, replacements)

        return text.strip()

    @staticmethod
    def _apply_replacements(text, replacements):
        """Apply a list of regex replacements to text."""
        for old, new in replacements:
            text = re.sub(old, new, text)
        return text

    def clean_fluency_transcripts(self, document, content):
        fluencyCleaner = FluencyCleaner(self.options)
        return fluencyCleaner.cleanFluency(document, content)



class FluencyCleaner:

    def __init__(self, options=None):
        self.options = options

    def cleanFluency(self, document, content):
        """Clean fluency task transcripts.
        
        Args:
            document: Document object containing metadata
            content: Text content to clean
        Returns:
            Cleaned text string
        """
        word_splitter = self.options['word_splitter']
        words = content.split(word_splitter)
        self.count_duplicates_and_hyphenated(document, words)

        content = re.sub(r'\s+', '', content).strip()
        words = [word for word in content.split(word_splitter) if word]

        # Apply cleaning operations based on options
        if self.options['remove_hyphens']:
            words = [word.replace('-', '') for word in words]
        if self.options['remove_duplicates']:
            words = self.remove_duplicates(words)
        if self.options['lowercase']:
            words = lowercase(words)

        return ' '.join(words)

    @staticmethod
    def remove_duplicates(words):

        # Remove duplicate words while preserving order
        word_counter = Counter(words)
        seen = set()
        cleaned_words = []

        for word in words:
            if word in seen and word_counter[word] > 1:
                word_counter[word] -= 1
            else:
                cleaned_words.append(word)
                seen.add(word)

        return cleaned_words

    @staticmethod
    def count_duplicates_and_hyphenated(document, words):
        word_counter = Counter(words)

        document.number_of_duplicates = sum(count - 1 for count in word_counter.values() if count > 1)
        document.number_of_hyphenated_words = sum(1 for word in words if '-' in word)


def _remove_brackets_and_bracketcontent(text):
    replacements = [
        (r'\(.*?\)', ''),
        (r'\<.*?\>', ''),
        (r'\[.*?\]', ''),
        (r'\{.*?\}', '')
    ]
    for old, new in replacements:
        text = re.sub(old, new, text)
    return text

def remove_timestamps(text, example):
    #removes timestamps with specified pattern
    pattern = _detect_timestamp_pattern(example)
    return re.sub(pattern, '', text)

def _detect_timestamp_pattern(example):
    pattern = re.escape(example)  # Escape special characters
    pattern = re.sub(r'\d', r'\\d', pattern)  # Replace digits with \d
    return pattern

def lowercase(text):
    if isinstance(text, str):
        return text.lower()
    elif isinstance(text, list):
        return [token.lower() for token in text]
    else:
        raise ValueError("Input to _lowercase must be either a string or a list of tokens")

def remove_punctuation(text):
    if isinstance(text, str):
        return text.translate(str.maketrans('', '', string.punctuation))
    elif isinstance(text, list):
        return [token.translate(str.maketrans('', '', string.punctuation)) for token in text]
    else:
        raise ValueError("Input to _remove_punctuation must be either a string or a list of tokens")

def _remove_special_characters(text, characters_to_remove):
    return text.translate(str.maketrans('', '', characters_to_remove))

def remove_speaker_tags(text, speaker_tags):
    pattern = re.compile(r'^(?:' + '|'.join(re.escape(tag) for tag in speaker_tags) + r'):\s*', re.MULTILINE)
    return re.sub(pattern, '', text)

def clean_subword_token_RoBERTa(token):
    """Clean RoBERTa subword tokens.
    
    Args:
        token: String token to clean
    Returns:
        Cleaned token or None if token should be ignored
    """
    # Special character mappings
    char_mappings = {
        '√§': 'ä',
        '√º': 'ü',
        '√∂': 'ö',
        '√í': 'í'
    }
    
    # Remove subword boundary marker
    clean_token = token.replace("▁", "")
    
    # Replace special characters
    for old, new in char_mappings.items():
        clean_token = clean_token.replace(old, new)
    
    # Remove bracketed content
    clean_token = re.sub(r"\[.*?\]|\(.*?\)", "", clean_token)
    
    # Keep only letters, numbers, and hyphens
    clean_token = re.sub(r"[^A-Za-z0-9\u00C0-\u017F\-]", "", clean_token)
    
    return None if clean_token.isdigit() else clean_token.strip()