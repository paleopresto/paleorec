import os
import sys
import io

from setuptools import setup, find_packages

version = '0.1.0'

# Read the readme file contents into variable
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='paleorec',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    version=version,
    license='Apache License 2.0',
    description='A Python package for paleoclimate recommendation system',
    long_description=read("README.md"),
    long_description_content_type = 'text/markdown',
    author='Shravya Manety, Deborah Khider',
    author_email='manety@usc.edu',
    url='https://github.com/paleopresto/paleorec',
    download_url='https://github.com/paleopresto/paleorec.git,
    keywords=['Paleoclimate, RecommendationSystem'],
    classifiers=[],
    install_requires=[
        "numpy>=1.18.5"
        "pandas>=1.0.5"
        "pytorch==1.7.1"
    ],
    python_requires=">=3.8.0"
)