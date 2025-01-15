from setuptools import setup, find_packages

setup(
    name='pelican',
    version='1.0.0',
    description='Preprocessing and Extraction of Linguistic Information for Computational Analysis',
    url='https://github.com/ypauli/PELICAN',
    author='Yves Pauli',
    author_email='yves.pauli@gmail.com',
    license='BSD 2-clause',
    packages=find_packages(),  # Automatically include all subpackages
    install_requires=[
        'numpy',
        'pandas',
        'pyyaml',
        'torch'
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
