import string
import re

class TextCleaner:
    def __init__(self, options):
        self.options = options

    def clean(self, text, characters_to_remove=None):

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
                (r'\s+([?.!"])',r'\1'),
                (r'\n\s*\n', '\n'),
                (r'\\', '')
            ]
            for old, new in replacements:
                text = re.sub(old, new, text)
        if self.options.get('remove_punctuation', True):
            text = self._remove_punctuation(text)
        if self.options.get('lowercase', True):
            text = text.lower()
        if characters_to_remove is not None:
            self._remove_special_characters(text, characters_to_remove)
        return text

    def _remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def _remove_special_characters(self, text, characters_to_remove):
        return text.translate(str.maketrans('', '', characters_to_remove))
