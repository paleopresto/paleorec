Creating Training and Test Data
===============================

The data collected from `The LinkedEarth Wiki <http://wiki.linked.earth>`_ and `LiPDverse <http://lipdverse.org>`_ have imbalance in data across each archiveType. Following is the distribution of the archives across the available compilations in LiPDverse.

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
This module lists a distribution of the archiveType in the input data.

Currently we are downsampling data for 'Wood' and 'MarineSediment'.

Routines
""""""""

.. toctree::
   :maxdepth: 1

   /creating_training_test_data/create_training_data.rst

Usage
"""""

To run the code execute the following command:

.. code-block:: none

   python create_training_data.py

Extensions
""""""""""

As more compilations are added, running this file will help in understanding the distribution of archiveTypes.
Manually select which archiveTypes need to be downsampled.

*Create method to handle this*
