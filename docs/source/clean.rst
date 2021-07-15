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

The current implementation offers the choice to consider files from the Linked Earth Wiki or to skip them.
In the case where we consider the Linked Earth Wiki files, we are considering the LiPDverse version of the file, if available, since it has been annotated and provides more information than the corresponding file on the Wiki.

Going forward, we might not want to consider the LiPD files from wiki. Please see the Usage instructions to only use the datasets provided through the command line.

The utils.readLipdFilesList python script will generate 2 pandas dataframe; first consisting of the proxyObservationType related information and second consisting of inferredVariableType related information. These 2 pandas dataframes are converted to csv and saved as **common_lipdverse_table_timestamp.csv** and **common_lipdverse_inferred_timestamp.csv**. After further processing, the two dataframes will be merged to generate a final csv file named **merged_common_lipdverse_inferred_timestamp.csv**

We are using the concept that inferredVariable can be predicted based on the string concatenation of interpretation/variable and the interpretation/variableDetail in a few cases. While scanning the LiPD file we hence generate 2 separte csv files; first consists of the predicted inferredVariable using interpretation/variable and interpretation/variableDetail;second consists of the inferredVariable information from the LiPD file itself. Since the second file doesn't contain the proxyObservationType information, we use dataframe manipulation to get the corresponding proxyObservationType information for the file and the archiveType and append it with the inferredVariableType information.

Routines
""""""""

.. toctree::
   :maxdepth: 1

   /cleaning_data/clean_data.rst

Usage
"""""

1. We already have the LiPD files from the wiki to start off the project.
2. Pages2k and Temp12k files are necessary for creating the training data:

      `Link for downloading PAGES2k dataset <https://lipdverse.org/Pages2kTemperature/current_version/PAGES2kv2.zip>`_.

      `Link for downloading Temp12k dataset <https://lipdverse.org/Temp12k/current_version/Temp12k1_0_1.zip>`_.

3. To provide other datasets, use the command  \'-o\' and provide comma-separated list of dataset paths.

      `Link for downloading ISO2k dataset <https://lipdverse.org/iso2k/current_version/iso2k1_0_0.zip>`_.

      `Link for downloading PalMod dataset <https://lipdverse.org/PalMod/current_version/PalMod1_0_1.zip>`_.

4. Please change the directory to \'cleaning_data\'

   To run the code, execute the following command:

   .. code-block:: none

      cd cleaning_data
      python clean_data.py -p [path-to-PAGES2k-dataset] -t [path-to-Temp12k-dataset] -o [path-to-dataset1],[path-to-dataset2]

5. You will be prompted asking if you would like to ignore files from the wiki:

   .. code-block:: none

      Please enter Y if you would like to ignore the wiki files:


Extensions
""""""""""

This module is created for the purpose of reading LiPD files from the provided datasets and extracting the required fields for the purpose of recommendation.
The 2 possible extensions are:

1. New files added to existing compilations
   Executing the clean_data.py script will read all the files within the datasets and generate new data to work with. Going forward, we would like the users to have an option to only read the additional files appended to the dataset and continue with the existing ones.

2. To extend this to other compilations.
   The important part here would be to take the input and read file names and store them in a list. This list will be read by the utils.readLipdFilesList module. This will not require any code changes.

3. To read more fields from the lipd file.
   This will require code changes to the utils.readLipdFilesList module.

Check your work
""""""""""""""""

After you executed the code, go to the `data/csv` and check that the follwing files have been added:
*common_lipdverse_table_timestamp.csv
*common_lipdverse_inferred_timestamp.csv
*merged_common_lipdverse_inferred_timestamp.csv

where  `timestamp` takes the form: mmddyyyy_hhmmss (e.g., 05252021_143300 for a file created on May 25th 2021 at 2:33pm).
