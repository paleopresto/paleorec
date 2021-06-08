Accuracy Calculation for the PaleoClimate Recommendation System
===============================================================

The metrics implemented for the recommendation system is 'Average Accuracy'.

The recommendation system returns the top 5 recommended values for a given fieldType in the sequence.
For a given test input sentence,
If the value in the test data is found to be the first recommended value, then score of 10 is returned for that prediction.
If the value in the test data is found to be in the top 5 recommended list, then score of 5 is returned for that prediction.
If the value in the test data is not found in the top 5 recommendation list, then score of 0 is returned.

Average of the prediction across the entire input is calculated and returned as the Average Accuracy for the test data.

Routines
--------

.. toctree::
   :caption: Accuracy Calculation for Models
   :maxdepth: 1

   /accuracy_calculation/fang_metrics.rst
   /accuracy_calculation/calc_accuracy_mc.rst


