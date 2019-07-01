import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

NAME = 'stoked'
DESCRIPTION = "Simulation and visualization of Stokesian dynamics for N interacting particles"
URL = ''
EMAIL = 'japarker@uchicago.edu'
AUTHOR = 'John Parker'
KEYWORDS = 'stokesian dynamics brownian'
# REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.3'
LICENSE = 'MIT'

REQUIRED = [
    'numpy', 
    'scipy',
    'matplotlib',
    'tqdm',
    'numpy_quaternion',
]

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    keywords=KEYWORDS,
    url=URL,
    packages=find_packages(),
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    install_requires=REQUIRED,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Science/Research',
    ],
    zip_safe=False,
)
