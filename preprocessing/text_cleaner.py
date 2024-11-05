import string
import re

class TextCleaner:
    def __init__(self, options):
        self.options = options

    def clean(self, text):
        if self.options.get('remove_brackets_and_bracketcontent', True):
            replacements = [
                (r'\(.*?\)', ''),
                (r'\<.*?\>', ''),
                (r'\[.*?\]', ''),
                (r'\{.*?\}', '')
            ]
            for old, new in replacements:
                text = re.sub(old, new, text)
        if self.options.get('general_cleaning', True):
            text = text.strip()
            replacements = [
                (' +', ' '),
                (r' \.', r'\.'),
                (r'\n\s*\n', '\n'),
                (r'\\', '')
            ]
            for old, new in replacements:
                text = re.sub(old, new, text)
        if self.options.get('remove_punctuation', True):
            text = self._remove_punctuation(text)
        if self.options.get('lowercase', True):
            text = text.lower()
        return text

    def _remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))
