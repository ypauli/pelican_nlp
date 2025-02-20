import re
from collections import Counter

class FluencyCleaner:

    def cleanFluency(self, document, content, word_splitter, remove_duplicates=True, clean_hyphens=True):

        self.count_duplicates_and_hyphenated(document, content.split(word_splitter))

        content = re.sub(r'\s+', '', content).strip()

        # Split and clean words
        words = [word for word in content.split(word_splitter) if word]

        if clean_hyphens:
            words = [word.replace('-', '') for word in words]

        if remove_duplicates:
            cleaned_words = self.remove_duplicates(words)
        else:
            cleaned_words = words

        return word_splitter.join(cleaned_words)

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


