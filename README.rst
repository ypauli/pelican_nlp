====================================
PELICAN-nlp
====================================

PELICAN-nlp stands for "Preprocessing and Extraction of Linguistic Information for Computational Analysis - Natural Language Processing".
This package is designed to enable the creation of standardized and reproducible language processing pipelines.
Input text from various tasks like discourse, fluency, image descriptions, etc. and extract linguistic features like semantic embeddings or logits.


.. image:: https://img.shields.io/pypi/v/package-name.svg
    :target: https://pypi.org/project/package-name/
    :alt: PyPI version

.. image:: https://img.shields.io/github/license/username/package-name.svg
    :target: https://github.com/ypauli/PELICAN/blob/main/LICENSE
    :alt: License

.. image:: https://img.shields.io/pypi/pyversions/package-name.svg
    :target: https://pypi.org/project/package-name/
    :alt: Supported Python Versions

Installation
============

You can install the package using pip:

.. code-block:: bash

    pip install pelican-nlp

For the latest development version:

.. code-block:: bash

    pip install git+https://github.com/ypauli/PELICAN.git

Usage
=====

Usage of pelican-nlp package:

.. code-block:: python

    import pelican-nlp as pelican

    configuration_file = /path/to/your/config/file
    pelican.run(configuration_file)

For pelican-nlp to work reliably data must be stored in the *Language Processing Data Structure (LPDS)* format.
*LPDS* is strongly inspired by the brain imaging data structure and provides conventions for storage and naming of text or audio files.

Text and audio files should follow the following naming convention:

subjectID_sessionID_task_task-supplement_corpus.extension

subjectID: ID of subject (e.g. sub-01), mandatory
sessionID: ID of session (e.g. ses-01), if available
task: task used for creation of file, mandatory
task-supplement: any additional information that the user wants to specify regarding the task, if available
corpus: (e.g. healthy-control / patient) specify all files that belong to the same group and should therefore be analyzed together with the same pipeline, mandatory
extension: file extension (e.g. txt / pdf / docx / rtf), mandatory

example filenames:
- sub-01_ses-01_interview_schizophrenia.rtf
- sub-03_ses-02_fluency_semantic_animals.docx


To optimize performance close other programs and limit gpu usage during language processing

Features
========

- Feature 1: Cleaning text files
    Includes handling of:
    - Whitespaces
    - Timestamps
    - Punctuation and special characters
    - Case-sensitivity

- Feature 2: Linguistic Feature Extraction
    Extract Linguistic Features:
    - Semantic embeddings
    - Logits
    - Distance from optimality
    - Semantic similarity

Examples
========

Here’s a more detailed usage example:

.. code-block:: python

    from package_name import SomeClass

    configuration_file = config_fluency.yml
    pelican.run(configuration_file)

*Link to config_fluency.yml*

Sample folder for data collection of semantic fluency task:
*Link to sample_folder*

Contributing
============

Contributions are welcome! Please check out the `contributing guide <https://github.com/ypauli/PELICAN/blob/main/CONTRIBUTING.md>`_.

License
=======

This project is licensed under Attribution-NonCommercial 4.0 International. See the `LICENSE <https://github.com/ypauli/PELICAN/blob/main/LICENSE>`_ file for details.
