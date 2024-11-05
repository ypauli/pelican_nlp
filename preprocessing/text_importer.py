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
