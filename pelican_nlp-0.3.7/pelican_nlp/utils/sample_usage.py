import pelican

file_path = 'your/file/path'

#return preprocessed transcript
preprocessed_files = pelican.preprocess(
    file_path=file_path,
    task=image_descriptions,
    general_cleaning=true,
    lowercase=true
)

#return embeddings from transcript
file_embeddings = pelican.extract_embeddings(
    file_path=file_path,
    mode="example_mode"
)