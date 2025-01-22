from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pelican-transcription",
    version="0.1.0",
    author="PUK Team",
    author_email="your.email@example.com",  # Update this
    description="A comprehensive audio transcription and editing tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pelican",  # Update this
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pelican-transcription=transcription.gui.transcription_gui:main",
        ],
    },
) 