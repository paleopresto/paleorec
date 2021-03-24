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

Going forward, we might not want to consider the LiPD files from wiki. Please see the Usage instructions to only use the datasets provided through the command line.

Using utils.readLipdFilesList will generate 2 pandas dataframe; first consisting of the proxyObservationType related information and second consisting of inferredVariableType related information. These 2 pandas dataframes are converted to csv and saved as common_lipdverse_table_timestamp.csv and common_lipdverse_inferred_timestamp.csv. After further processing, the two dataframes will be merged to generate a final csv file named merged_common_lipdverse_inferred_timestamp.csv

Routines
""""""""

.. toctree::
   :maxdepth: 1

   /cleaning_data/clean_data.rst

Usage
"""""

1. We already have the LiPD files from the wiki to start off the project.
2. Pages2k and Temp12k files are necessary for creating the training data:

      `Link for downloading PAGES2k dataset <http://lipdverse.org/Pages2kTemperature/current_version/>`_.

      `Link for downloading Temp12k dataset <http://lipdverse.org/Temp12k/current_version/>`_.

3. To provide other datasets, use the command  \'-o\' and provide comma-separated list of dataset paths.
4. Please change the directory to \'cleaning_data\'

   To run the code execute the following command:

   .. code-block:: none
   
      cd cleaning_data
      python clean_data.py -p [path-to-PAGES2k-dataset] -t [path-to-Temp12k-dataset] -o [path-to-dataset1],[path-to-dataset2]

5. You will be prompted asking if you would like to ignore files from the wiki: 
   
   .. code-block:: none
      
      Please enter Y if you would like to ignore the wiki files:    
   

Extensions
""""""""""

This module is created for the purpose of reading wiki files or corresponding versions in LiPDverse and extracting the required field for the recommendation purpose. 
The 2 possible extensions are:

1. To extend this to other compilations.
   The important part here would be to take the input and read file names and store them in a list. This list will be read by the utils.readLipdFilesList module.

2. To read more fields from the lipd file.
   This will require code changes to the utils.readLipdFilesList module.

