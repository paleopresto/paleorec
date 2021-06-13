Creating Training and Test Data
===============================

The data collected from `The LinkedEarth Wiki <http://wiki.linked.earth>`_ and `LiPDverse <http://lipdverse.org>`_ have imbalance in data across each archiveType. Following is the distribution of the archives across the available compilations in LiPDverse.

Data As of 03/19/2021

PAGES2k::

    {'speleothem': 5,
    'lake sediment': 37,
    'glacier ice': 70,
    'tree': 1777,
    'marine sediment': 31,
    'hybrid': 1,
    'documents': 1,
    'coral': 141,
    'bivalve': 1,
    'sclerosponge': 6}

Temp12k::

    {'MarineSediment': 62,
    'GlacierIce': 15,
    'LakeSediment': 23,
    'Ice-other': 7,
    'Speleothem': 4,
    'Midden': 27,
    'Peat': 2}

iso2k::

    {'Coral': 52,
    'GroundIce': 8,
    'GlacierIce': 198,
    'LakeSediment': 132,
    'MarineSediment': 62,
    'MolluskShells': 1,
    'TerrestrialSediment': 8,
    'Speleothem': 56,
    'Sclerosponge': 4,
    'Wood': 68}

PalMod::

    {'marine sediment': 924}

As more compilations are added to LiPDverse, this distribution will change. Since we are modeling the recommendation system as a sequential prediction model, the training data should contain nearly equal number of samples for each archive to have an unbiased model.
To balance out the disribution of the archiveTypes, we downsample the data for the archiveTypes which have abundant samples.

The input to this module is the latest file 'merged_common_lipdverse_inferred_timestamp.csv' created by the clean_data.py module.

There are many proxyObservationTypes, interpretation/variable and interpretation/variableDetail that appear only a few times. Since they do not contribute heavily to the recommendation system, we considered it best to remove them from the data as they are outliers. The user is presented with the count of samples for each of the proxyObservationType. The user is then requested for a number \'k\' to eliminate any-co-k occurances in the data. Similarly user is requested to enter the value of 'k' for interpretation/variable and interpretation/variableDetail.

Running this module will list the number of samples for each archiveType in the input data. The user is requested to enter a comma-separated string of number for each archiveType they wish to downsample. This is followed by a request to enter a numeric value to which each archiveType needs to be downsampled to.

Routines
""""""""

.. toctree::
   :maxdepth: 1

   /creating_training_test_data/create_training_test_data.rst

Usage
"""""
Please change the directory to \'creating_training_test_data\'
To run the code execute the following command:

.. code-block:: none

    cd creating_train_test_data
    python create_training_test_data.py

    Please enter the value of 'k' to replace any-co-k instances : 5

    Count for each instance of Archive Type:
    1. Speleothem = 159
    2. LakeSediment = 538
    3. GlacierIce = 487
    4. Wood = 1942
    5. MarineSediment = 1573
    6. Hybrid = 2
    7. Documents = 2
    8. Coral = 314
    9. MolluskShell = 3
    10. Sclerosponge = 16
    11. Peat = 8
    12. Ice-other = 16
    13. Midden = 54
    14. GroundIce = 22
    15. TerrestrialSediment = 34

    Using the item number beside the archive type, list the archive types to be downsampled separated by ',' or enter none: 4,5
    Please enter the numeric value to downsampled the above list of Archive Types in same order :350,350

Extensions
""""""""""

As more compilations are added, running this file will help in understanding the distribution of archiveTypes.

Expert advise from the users of LiPD data is required to complete this module. User input is required to eliminate any-co-k values from proxyObservationType, interpretation/variable and interpretation/variableDetail.

Apart from this user needs to ensure that the data is class-balanced by downsampling the archiveTypes that have abundant samples.

Check your work
"""""""""""""""

You will be creating two files that can be found in `data/csv`:
* lipdverse_downsampled_timestamp.csv
* lipdverse_test_timestamp.csv

You will be creating one file that can be found in `data/autocomplete`:
* autocomplete_file_timestamp.csv

You will be creating one file that can be found in `data/ground_truth_info`:
* ground_truth_label_timestamp.csv

where  `timestamp` takes the form: mmddyyyy_hhmmss (e.g., 05252021_143300 for a file created on May 25th 2021 at 2:33pm).
