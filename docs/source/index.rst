.. PaleoRec documentation master file, created by
   sphinx-quickstart on Thu Mar 11 13:19:31 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PaleoRec
========

PaleoRec is a recommender system for paleoclimate data based on the LiPD format. 

The requirement:
`lipd.net <https://lipd.net/playground>`_ playground enables climate scientists and users of the LiPD data format to create, edit and upload lipd files. However, most of the users enter custom values for the same data causing large variations in the data.

The effort:
In an effort to simplify the process of creating LiPD files and maintaining consistency across all the files we have developed PaleoRec. It is a simple recommendation system accompanied by autocomplete suggestions. The recommendations are provided for measurement tables within paleoData field of the LiPD files.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   clean.rst
   create_train_test.rst
   train.rst
   predict.rst
   accuracy_calc.rst

Dependencies
============

Dependencies for PaleoRec are `here <https://github.com/paleopresto/paleorec/blob/main/environment.yml>`_.

Repo Details
============

The command to link to git repo is `here <https://github.com/paleopresto/paleorec>`_.

Or `click here <https://github.com/paleopresto/paleorec/archive/refs/heads/main.zip>`_ to directly download the zip file for the repo 
