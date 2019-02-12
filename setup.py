import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "pedesis",
    version = "0.1.0",
    author = "John Parker",
    author_email = "japarker@uchicago.com",
    description = ("Simulation and visualization of Brownian motion and related dynamics"),
    license = "MIT",
    keywords = "",
    packages=['pedesis'],
    long_description=read('README.md'),
    install_requires=['numpy', 'scipy'],
    include_package_data = True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
    ],
)
