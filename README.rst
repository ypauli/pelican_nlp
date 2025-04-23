====================================
pelican_nlp
====================================

.. |logo| image:: docs/images/pelican_logo.png
    :alt: PELICAN_nlp Logo
    :width: 200px

+------------+-------------------------------------------------------------------+
| |logo|     | pelican_nlp stands for "Preprocessing and Extraction of Linguistic|
|            | Information for Computational Analysis - Natural Language         |
|            | Processing". This package enables the creation of standardized and|
|            | reproducible language processing pipelines, extracting linguistic |
|            | features from various tasks like discourse, fluency, and image    |
|            | descriptions.                                                     |
+------------+-------------------------------------------------------------------+

.. image:: https://img.shields.io/pypi/v/package-name.svg
    :target: https://pypi.org/project/pelican_nlp/
    :alt: PyPI version

.. image:: https://img.shields.io/github/license/username/package-name.svg
    :target: https://github.com/ypauli/pelican_nlp/blob/main/LICENSE
    :alt: License

.. image:: https://img.shields.io/pypi/pyversions/package-name.svg
    :target: https://pypi.org/project/pelican_nlp/
    :alt: Supported Python Versions

Installation
============

Create conda environment

.. code-block:: bash

    conda create -n pelican-nlp -c defaults python=3.10

Activate environment

.. code-block:: bash

    conda activate pelican-nlp

Install the package using pip:

.. code-block:: bash

    pip install pelican_nlp

For the latest development version:

.. code-block:: bash

    pip install https://github.com/ypauli/pelican_nlp/releases/tag/v0.1.2-alpha

Usage
=====

To run pelican_nlp you need a configuration.yml file in your project directory, which specifies the configurations used for your project.
Sample configuration files can be found on the pelican_nlp github repository: https://github.com/ypauli/pelican_nlp/tree/main/sample_configuration_files

Adapt your configuration file to your needs and save your personal configuration.yml file to your main project directory.

Running pelican_nlp with your configurations can be done directly from the command line interface or via Python script.

Run from command line:

Navigate to main project directory in command line and enter the following command (Note: Folder must contain your subjects folder and your configuration.yml file):

.. code-block:: bash

    conda activate pelican-nlp
    pelican-run


Run with python script:

Create python file with IDE of your choice (e.g. Visual Studio Code, Pycharm, etc.) and copy the following code into the file:
Make sure to use the previously created conda environment 'pelican-nlp' for your project.

Run the following Python code:
.. code-block:: python

    from pelican_nlp.main import Pelican

    configuration_file = "/path/to/your/config/file.yml"
    pelican = Pelican(configuration_file)
    pelican.run()

Replace "/path/to/your/config/file" with the path to your configuration file located in your main project folder.

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

You can find example setups on the github repository in the `examples <https://github.com/ypauli/pelican_nlp/tree/main/examples>`_ folder:

Contributing
============

Contributions are welcome! Please check out the `contributing guide <https://github.com/ypauli/pelican_nlp/blob/main/CONTRIBUTING.md>`_.

License
=======

This project is licensed under Attribution-NonCommercial 4.0 International. See the `LICENSE <https://github.com/ypauli/pelican_nlp/blob/main/LICENSE>`_ file for details.
