import string
import re
from collections import Counter

class TextCleaner:
    def __init__(self, options=None):
        self.options = options

    def clean(self, document, text, characters_to_remove=None):

        if self.options.get('remove_timestamps', True):
            text = self.remove_timestamps(text, self.options.get('timestamp_pattern_example'))

        if characters_to_remove is not None:
            text = self._remove_special_characters(text, characters_to_remove)

        if self.options.get('fluency_task'):
            text = self.clean_fluency_transcripts(document, text)

        if self.options.get('general_cleaning', True):
            replacements = [
                (r'/', ''),
                (r'\s+([?.!,"])',r'\1'),
                (r'\n\s*\n', '\n'),
                (r'\\', ''),
                (' +', ' ')
            ]
            for old, new in replacements:
                text = re.sub(old, new, text)
            text = text.strip()

        return text

    def clean_fluency_transcripts(self, document, content):
        fluencyCleaner = FluencyCleaner()
        return fluencyCleaner.cleanFluency(document, content, self.options.get('word_splitter'), self.options.get('remove_duplicates'), self.options.get('remove_hyphens'))

    def remove_timestamps(self, text, example):
        #removes timestamps with specified pattern
        pattern = self._detect_timestamp_pattern(example)
        return re.sub(pattern, '', text)

    @staticmethod
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

    @staticmethod
    def _detect_timestamp_pattern(example):
        pattern = re.escape(example)  # Escape special characters
        pattern = re.sub(r'\d', r'\\d', pattern)  # Replace digits with \d
        return pattern

    @staticmethod
    def _lowercase(text):
        if isinstance(text, str):
            return text.lower()
        elif isinstance(text, list):
            return [token.lower() for token in text]
        else:
            raise ValueError("Input to _lowercase must be either a string or a list of tokens")

    @staticmethod
    def _remove_punctuation(text):
        if isinstance(text, str):
            return text.translate(str.maketrans('', '', string.punctuation))
        elif isinstance(text, list):
            return [token.translate(str.maketrans('', '', string.punctuation)) for token in text]
        else:
            raise ValueError("Input to _remove_punctuation must be either a string or a list of tokens")

    @staticmethod
    def _remove_special_characters(text, characters_to_remove):
        return text.translate(str.maketrans('', '', characters_to_remove))

    @staticmethod
    def remove_speaker_tags(text, speaker_tags):
        pattern = re.compile(r'^(?:' + '|'.join(re.escape(tag) for tag in speaker_tags) + r'):\s*', re.MULTILINE)
        return re.sub(pattern, '', text)

    @staticmethod
    def clean_subword_token_RoBERTa(token):

        # Remove the '▁' prefix which indicates subword boundaries (for subwords, keep as is)
        clean_token = token.replace("▁", "")  # The '▁' symbol represents space in subword tokenization

        # Handle special character encoding issues (e.g., '√§' -> 'ä', '√º' -> 'ü', etc.)
        clean_token = clean_token.replace('√§', 'ä').replace('√º', 'ü').replace('√∂', 'ö').replace('√í', 'í')

        clean_token = re.sub(r"\[.*?\]", "", clean_token)  # Remove any text inside square brackets
        clean_token = re.sub(r"\(.*?\)", "", clean_token)  # Remove any text inside parentheses

        # Remove unwanted punctuation or symbols that aren't useful for fusion
        clean_token = re.sub(r"[^A-Za-z0-9\u00C0-\u017F\-]", "", clean_token)  # Keep only letters, numbers, and hyphens

        # Remove numbers (unless part of meaningful words)
        if clean_token.isdigit():
            return None  # Ignore speaker labels and standalone numbers

        return clean_token.strip()  # Ensure there are no extra spaces

class FluencyCleaner:

    def __init__(self, options=None):
        self.options = options

    def cleanFluency(self, document, content, remove_duplicates=True, clean_hyphens=True):

        word_splitter = self.options['word_splitter']
        self.count_duplicates_and_hyphenated(document, content.split(word_splitter))

        content = re.sub(r'\s+', '', content).strip()

        # Split and clean words
        words = [word for word in content.split(word_splitter) if word]

        if clean_hyphens:
            words = [word.replace('-', '') for word in words]

        if remove_duplicates:
            words = self.remove_duplicates(words)

        return f'{word_splitter} '.join(words)

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