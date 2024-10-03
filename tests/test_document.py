#unit tests for document

import unittest
from preprocessing.text_document import TextDocument

class TestTextDocumentDialog(unittest.TestCase):
    def test_dialog_extraction(self):
        raw_text = '''
        John: Hey, how are you?
        Mary: I'm good, thanks! How about you?
        "This is a thought," she said.
        "Yes," he replied.
        '''
        document = TextDocument('dialog.txt')
        document.raw_text = raw_text

        # Clean with dialog extraction
        spoken_text = document.extract_spoken_text(document.raw_text)

        # Expected spoken lines combined from both character names and quotes
        expected_text = "Hey, how are you? I'm good, thanks! How about you? This is a thought Yes"

        self.assertEqual(spoken_text, expected_text)


if __name__ == '__main__':
    unittest.main()
