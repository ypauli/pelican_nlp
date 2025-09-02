import os

class TextImporter:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_text(self, file_path):
        # Possible file formats txt and docx, expand if necessary
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == '.txt':
            return self._load_txt(file_path)
        elif ext == '.docx':
            return self._load_docx(file_path)
        elif ext == '.rtf':
            return self._load_rtf(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _load_txt(self,file_path):
        with open(file_path, 'r') as file:
            return file.read()

    def _load_docx(self,file_path):
        import docx2txt
        doc = docx2txt.process(file_path)
        return doc
        #return '\n'.join([para.text for para in doc.paragraphs])

    def _load_rtf(self, file_path):
        """Read RTF file and convert its content to plain text."""
        from striprtf.striprtf import rtf_to_text
        import chardet
        with open(file_path, "rb") as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result["encoding"]

        with open(file_path, "r", encoding=encoding, errors="ignore") as file:
            rtf_content = file.read()

        return rtf_to_text(rtf_content)