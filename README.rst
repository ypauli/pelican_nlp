====================================
PELICAN_nlp
====================================

PELICAN_nlp stands for "Preprocessing and Extraction of Linguistic Information for Computational Analysis - Natural Language Processing". This package enables the creation of standardized and reproducible language processing pipelines, extracting linguistic features from various tasks like discourse, fluency, and image descriptions.

.. image:: https://img.shields.io/pypi/v/package-name.svg
    :target: https://pypi.org/project/pelican-nlp/
    :alt: PyPI version

.. image:: https://img.shields.io/github/license/username/package-name.svg
    :target: https://github.com/ypauli/PELICAN-nlp/blob/main/LICENSE
    :alt: License

.. image:: https://img.shields.io/pypi/pyversions/package-name.svg
    :target: https://pypi.org/project/pelican-nlp/
    :alt: Supported Python Versions

Installation
============

Install the package using pip:

.. code-block:: bash

    pip install pelican_nlp

For the latest development version:

.. code-block:: bash

    pip install https://github.com/yourusername/yourrepo/releases/tag/v0.1.0-alpha

Usage
=====

To use the pelican_nlp package:

Adapt your configuration file to your needs.
ALWAYS change the specified project folder location.

.. code-block:: python

    from pelican_nlp.main import Pelican

    configuration_file = "/path/to/your/config/file"
    pelican = Pelican(configuration_file)
    pelican.run()

For reliable operation, data must be stored in the *Language Processing Data Structure (LPDS)* format, inspired by brain imaging data structure conventions.

Text and audio files should follow this naming convention:

[subjectID]_[sessionID]_[task]_[task-supplement]_[corpus].[extension]

- subjectID: ID of subject (e.g., sub-01), mandatory
- sessionID: ID of session (e.g., ses-01), if available
- task: task used for file creation, mandatory
- task-supplement: additional information regarding the task, if available
- corpus: (e.g., healthy-control / patient) specify files belonging to the same group, mandatory
- extension: file extension (e.g., txt / pdf / docx / rtf), mandatory

Example filenames:

- sub-01_interview_schizophrenia.rtf
- sub-03_ses-02_fluency_semantic_animals.docx

To optimize performance, close other programs and limit GPU usage during language processing.

Features
========

- **Feature 1: Cleaning text files**
    - Handles whitespaces, timestamps, punctuation, special characters, and case-sensitivity.

- **Feature 2: Linguistic Feature Extraction**
    - Extracts semantic embeddings, logits, distance from optimality, and semantic similarity.

Examples
========

You can find example setups in the [`examples/`](https://github.com/ypauli/PELICAN-nlp/examples) folder.
ALWAYS change the path to the project folder specified in the configuration file to your specific project location.

Contributing
============

Contributions are welcome! Please check out the `contributing guide <https://github.com/ypauli/PELICAN-nlp/blob/main/CONTRIBUTING.md>`_.

License
=======

This project is licensed under Attribution-NonCommercial 4.0 International. See the `LICENSE <https://github.com/ypauli/PELICAN-nlp/blob/main/LICENSE>`_ file for details.
