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

**From this repository (default for development and pre-PyPI testing).**
This makes ``pelican-run`` use the code on disk, not a published wheel:

.. code-block:: bash

    cd /path/to/PELICAN-nlp
    pip install -e '.[dev]'

Confirm the command is bound to the checkout (run this from any directory, not only the repo):

.. code-block:: bash

    python -c "import pelican_nlp.cli; print(pelican_nlp.cli.__file__)"

The printed path should be ``.../PELICAN-nlp/pelican_nlp/cli.py``. If it contains ``site-packages``, the env is still on an installed wheel; rerun the editable install.

**From PyPI (released package only):**

.. code-block:: bash

    pip install 'pelican_nlp[dev]'

Optional extras
---------------

``pip install pelican_nlp`` still installs the full stack. Extra names document
which libraries a YAML config needs. They do **not** shrink that default
install today: ``pip install 'pelican_nlp[transcription]'`` is not smaller
while torch, fastText, and the other heavy libraries remain required
dependencies. The names are the install contract if the default is slimmed
later.

.. code-block:: bash

    pip install 'pelican_nlp[transcription]'
    pip install 'pelican_nlp[embeddings]'
    pip install 'pelican_nlp[all]'

================== ===============================================================
Extra              YAML that needs it
================== ===============================================================
``transcription`` ``input_file: audio`` with a ``transcription:`` block
``acoustic``       ``opensmile_feature_extraction`` / ``prosogram_extraction``
``embeddings``     ``metrics_to_extract`` embeddings, logits, or perplexity
``nlp``            ``pipeline_options.normalize_text``
``topic``          ``metrics_to_extract`` includes ``topic_modeling``
``all``            union of the extras above (includes BERTopic)
``dev``            pytest (``pelican-run --run-tests``)
================== ===============================================================

Usage
=====

To run ``pelican_nlp``, you need a ``configuration.yml`` file in your main project directory. This file defines the settings and parameters used for your project.

Sample configuration files are available here:
`https://github.com/ypauli/pelican_nlp/tree/main/examples <https://github.com/ypauli/pelican_nlp/tree/main/examples>`_

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

Running tests
-------------

After the editable install above, run the suite from any directory (same conda env):

.. code-block:: bash

    pelican-run --run-tests

That uses local ``tests/`` and ``examples/``. You do not need to publish to PyPI first.

Include example golden tests (needs models; the same set as ``pytest --run-examples``):

.. code-block:: bash

    pelican-run --run-tests --examples
    pelican-run --run-tests --examples fluency,discourse

Example runs print a short progress summary. For the full pipeline dump:

.. code-block:: bash

    pelican-run --run-tests --examples --example-logs

Forward extra pytest flags after ``--``:

.. code-block:: bash

    pelican-run --run-tests -- -k test_model_registry -q

``pelican-test-examples`` is an alias for ``pelican-run --run-tests --examples``.

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
    - Extracts semantic embeddings, logits, distance from optimality, perplexity and semantic similarity.

- **Feature 3: Acoustic Feature Extraction**
    - Extracts prosogram and openSMILE feature.

Examples
========

You can find example setups on the github repository in the `examples <https://github.com/ypauli/pelican_nlp/tree/main/examples>`_ folder:

Contributing
============

Contributions are welcome! Please check out the `contributing guide <https://github.com/ypauli/pelican_nlp/blob/main/CONTRIBUTING.md>`_.

License
=======

This project is licensed under Attribution-NonCommercial 4.0 International. See the `LICENSE <https://github.com/ypauli/pelican_nlp/blob/main/LICENSE>`_ file for details.

Citation
========

If you use this project, please cite:

Pauli Y, Marsman J-B, Rabe F, et al. Standardising the NLP Workflow: A Framework for Reproducible Linguistic Analysis. arXiv preprint arXiv:2511.15512 [cs.CL] 2025.
https://doi.org/10.48550/arXiv.2511.15512
