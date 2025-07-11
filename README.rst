====================================
pelican_nlp
====================================

.. |logo| image:: https://raw.githubusercontent.com/ypauli/pelican_nlp/main/docs/images/pelican_logo.png
    :alt: pelican_nlp Logo
    :width: 200px

+------------+-------------------------------------------------------------------+
| |logo|     | pelican_nlp stands for "Preprocessing and Extraction of Linguistic|
|            | Information for Computational Analysis - Natural Language         |
|            | Processing". This package enables the creation of standardized and|
|            | reproducible language processing pipelines, extracting linguistic |
|            | features from various tasks like discourse, fluency, and image    |
|            | descriptions.                                                     |
+------------+-------------------------------------------------------------------+

.. image:: https://img.shields.io/pypi/v/pelican_nlp.svg
    :target: https://pypi.org/project/pelican_nlp/
    :alt: PyPI version

.. image:: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
    :target: https://github.com/ypauli/pelican_nlp/blob/main/LICENSE
    :alt: License CC BY-NC 4.0

.. image:: https://img.shields.io/pypi/pyversions/pelican_nlp.svg
    :target: https://pypi.org/project/pelican_nlp/
    :alt: Supported Python Versions

.. image:: https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg
    :target: https://github.com/ypauli/pelican_nlp/blob/main/CONTRIBUTING.md
    :alt: Contributions Welcome

Installation
============

Create conda environment

.. code-block:: bash

    conda create --name pelican-nlp --channel defaults python=3.10

Activate environment

.. code-block:: bash

    conda activate pelican-nlp

Install the package using pip:

.. code-block:: bash

    pip install pelican-nlp

For the latest development version:

.. code-block:: bash

    pip install https://github.com/ypauli/pelican_nlp/releases/tag/v0.1.2-alpha

Usage
=====

To run ``pelican_nlp``, you need a ``configuration.yml`` file in your main project directory. This file defines the settings and parameters used for your project.

Sample configuration files are available here:
`https://github.com/ypauli/pelican_nlp/tree/main/sample_configuration_files <https://github.com/ypauli/pelican_nlp/tree/main/sample_configuration_files>`_

1. Adapt a sample configuration to your needs.
2. Save your personalized ``configuration.yml`` in the root of your project directory.

Running pelican_nlp
-------------------

You can run ``pelican_nlp`` via the command line or a Python script.

**From the command line**:

Navigate to your project directory (must contain your ``participants/`` folder and ``configuration.yml``), then run:

.. code-block:: bash

    conda activate pelican-nlp
    pelican-run

To optimize performance, close other programs and limit GPU usage during language processing.

Data Format Requirements: LPDS
------------------------------

For reliable operation, your data must follow the *Language Processing Data Structure (LPDS)*, inspired by brain imaging data structures like BIDS.

Main Concepts (Quick Guide)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Project Root**: Contains a ``participants/`` folder plus optional files like ``participants.tsv``, ``dataset_description.json``, and ``README``.
- **Participants**: Each participant has a folder named ``part-<ID>`` (e.g., ``part-01``).
- **Sessions (Optional)**: For longitudinal studies, use ``ses-<ID>`` subfolders inside each participant folder.
- **Tasks/Contexts**: Each session (or directly in the participant folder for non-longitudinal studies) includes subfolders for specific tasks (e.g., ``interview``, ``fluency``, ``image-description``).
- **Data Files**: Named with structured metadata, e.g.:
  ``part-01_ses-01_task-fluency_cat-semantic_acq-baseline_transcript.txt``

Filename Structure
~~~~~~~~~~~~~~~~~~

Filenames follow this format::

    part-<id>[_ses-<id>]_task-<label>[_<key>-<value>...][_suffix].<extension>

- **Required Entities**: ``part``, ``task``
- **Optional Entities Examples**: ``ses``, ``cat``, ``acq``, ``proc``, ``metric``, ``model``, ``run``, ``group``, ``param``
- **Suffix Examples**: ``transcript``, ``audio``, ``embeddings``, ``logits``, ``annotations``

Example Project Structure
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    my_project/
    ├── participants/
    │   ├── part-01/
    │   │   └── ses-01/
    │   │       └── interview/
    │   │           └── part-01_ses-01_task-interview_transcript.txt
    │   └── part-02/
    │       └── fluency/
    │           └── part-02_task-fluency_audio.wav
    ├── configuration.yml
    ├── dataset_description.json
    ├── participants.tsv
    └── README.md


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
