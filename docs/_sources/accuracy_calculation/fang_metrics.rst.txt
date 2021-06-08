Evaluation Metrics for LSTM
===========================

Functions
---------

**get_mrr_score(grnd_truth_i,hxi):**
    
    Formula from https://arxiv.org/pdf/1808.06414.pdf 'Next Item Recommendation with Self-Attention' pg 6



**get_recall_score(grnd_truth_i, hxi):**

    Formula from https://arxiv.org/pdf/1808.06414.pdf 'Next Item Recommendation with Self-Attention' pg 5 


**mean_recall(val):**

    Method to calculate the average recall for all the examples in the test data for a particular chain length across all the recommendation set size.(3, 5, 7, 10 ,12, 14, 16)

    Parameters:

    val : int
        Chain length across which mean is calculated.

    Returns:

    res : list
        List containing the average recall across a particular chain size for different recommendation set sizes.

**mean_mrr(val):**

    Method to calculate the MRR for a chain sizes across different recommendation set sizes.(3, 5, 7, 10)

    Parameters:

    val : int
        Chain length across which mean is calculated

    Returns:

    res : list
        list containing the average MRR across a particular chain length for different recommendation set sizes.


**calculate_score_for_test_data_chain():**

    Calculates the evaluation metrics for the provided test data by generating different length chains from the input.
    Generates a line chart of the average recall and MRR across different recommednation set sizes.

    Returns:

    None.

Usage
-----

To run the code execute the following command:

.. code-block:: none

    cd /accuracy_calculation/lstm

1. Open the **run_fang_metrics.ipynb** file.
2. Choose 'Kernel' drop down and the 'Restart and Run all' option within it.
3. The output will show you the line plot for Recall and MRR for different chain lengths across different recommendation set sizes.

