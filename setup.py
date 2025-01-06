from setuptools import setup

setup(
    name='pelicanlang',
    version='1.0.0',
    description='Preprocessing and Extraction of Linguistic Information for Computational Analysis',
    url='https://github.com/ypauli/PELICAN',
    author='Yves Pauli',
    author_email='yves.pauli@gmail.com',
    license='BSD 2-clause',
    packages=['pelican'],
    install_requires=['numpy',
                      'pandas'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)