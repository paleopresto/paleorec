Cleaning the data
=================

PaleoRec uses data from `The LinkedEarth Wiki <http://wiki.linked.earth>`_ and `LiPDverse <http://lipdverse.org>`_.

Current focus has been on extracting information for the following fields :
archiveType, proxyObservationType, units, interpretation/variable, interpretation/VariableDetail, inferredVariable, inferredVariableUnits

To ensure that we were utilizing all the available data, we used the data queried from the Wiki using SPARQL queries. Apart from this, the data curated in the `Linked Earth Ontology <http://linked.earth/ontology/>`_ for Paleoclimate Data is used.
Cleaning the data involved creating a mapping of the incorrect input to its correct value.
Examples - 
Spelling errors, repeated values like Temperature1, Temperature2, Depth, Depth-cm, Mg_Ca
Incorrect Value  - d18o -> D18O, D180 -> D18O

For the current implementation we are considering the LiPDverse version of the file if available else we will consider the corresponding file from LinkedEarth Wiki.

Using utils.readLipdFilesList will generate 2 pandas dataframe; first consisting of the proxyObservationType related information and second consisting of inferredVariableType related information. These 2 pandas dataframes are converted to csv and saved as common_lipdverse_table_timestamp.csv and common_lipdverse_inferred_timestamp.csv. After further processing, the two dataframes will be merged to generate a final csv file named merged_common_lipdverse_inferred_timestamp.csv

Routines
""""""""

.. toctree::
   :maxdepth: 1

   /cleaning_wiki_data/clean_data.rst

Usage
"""""

We already have the LiPD files from the wiki to start off the project.
To provide the files from LiPDverse, please download the files from `LiPDverse <http://lipdverse.org>`_.
All the commandline arguments with '-' are optional.
To run the code execute the following command:

.. code-block:: none

   python clean_data.py -p [path-to-PAGES2k-dataset] -t [path-to-Temp12k-dataset] -i [path-to-iso2k-dataset] -pm [path-to-Palmod-dataset]

   OR

   python clean_data.py -pages2k [path-to-PAGES2k-dataset] -temp12k [path-to-Temp12k-dataset] -iso2k [path-to-iso2k-dataset] -palmod [path-to-Palmod-dataset]

Please provide the path where you have downloaded the files on your machine for each of the datasets.

Extensions
""""""""""

This module is created for the purpose of reading wiki files or corresponding versions in LiPDverse and extracting the required field for the recommendation purpose. 
To extend it to other compilations we will require code changes to enable reading all files.
The important part here would be to take the input and read file names and store them in a list. This list will be read by the utils.readLipdFilesList module.
These code changes will be a part of the future release.
