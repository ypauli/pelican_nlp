import string
import re

class TextCleaner:
    def __init__(self, options=None):
        self.options = options

    def clean(self, text, characters_to_remove=None, num_speakers=None):

        if self.options.get('remove_timestamps', True):
            text = self.remove_timestamps(text, self.options.get('timestamp_pattern_example'))

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
                (r'/', ''),
                (r'…', ''),
                (r'\.{3}', ''),
                (r'\s+([?.!,"])',r'\1'),
                (r'\n\s*\n', '\n'),
                (r'\\', ''),
                (' +', ' ')
            ]
            for old, new in replacements:
                text = re.sub(old, new, text)

        if characters_to_remove is not None:
            self._remove_special_characters(text, characters_to_remove)
        return text

    def _detect_timestamp_pattern(self, example):
        pattern = re.escape(example)  # Escape special characters
        pattern = re.sub(r'\d', r'\\d', pattern)  # Replace digits with \d
        return pattern

    def remove_timestamps(self, text, example):
        #removes timestamps with specified pattern
        pattern = self._detect_timestamp_pattern(example)
        return re.sub(pattern, '', text)

    def _lowercase(self, text):
        return text.lower()

    def _remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def _remove_special_characters(self, text, characters_to_remove):
        return text.translate(str.maketrans('', '', characters_to_remove))

    def clean_text_diarization_all(self, text, stopwords_list, remove_numbers=False):

        #This function is outdated and just like it is found in original diarization script. Do not use as is.

        """Clean and preprocess text."""
        text = re.sub(r"^[A-Z][0-9]?::?\s*", "", text, flags=re.MULTILINE)
        text = text.replace("\xa0", " ")
        text = re.sub(r'[\'"`]', "", text)
        text = text.replace("\\", "").replace("/", "")
        if remove_numbers:
            text = re.sub(r"\b\d+\w*|\w*\d+\w*", "", text)
        text = re.sub(r"\(.*?\)|\{.*?\}", "", text)
        pattern = r"\[\s*(?:\d+\s*)?\s*(.*?)\s*(?:\d+\\s*)?\s*\]"
        text = re.sub(pattern, r"\1", text)

        sentences = sent_tokenize(text)
        cleaned_sentences = []
        for sentence in sentences:
            words = [word for word in word_tokenize(sentence) if word.isalnum()]
            filtered_words = [word for word in words if word not in stopwords_list]
            if len(filtered_words) > 1:
                cleaned_sentence = " ".join(filtered_words)
                cleaned_sentences.append(cleaned_sentence)

        cleaned_text = ". ".join(cleaned_sentences) + "."
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        return cleaned_text
