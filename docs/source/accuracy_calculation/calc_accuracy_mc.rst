Accuracy Calculation for Markov Chains
======================================

Functions
---------

**getScoreForResult(test_val, result_list):**
    
    Calculate Accuracy Score for Markov Chain prediction. 
    The function returns 10 if the actual value from input matches the 1st string in the list of top 5 predictions using Markov Chains.
    Else it returns 5 if the actual value is present in the list of top 5 predictions using Markov Chains.
    Else it returns 0.
    
    Parameters:

    test_val : string
        Actual value for the test input.

    result_list : list
        List consisting of the predictions using Markov Chains.

    Returns:

    int
        Accuracy score depending on where the actual value is present in list of predicted values.
    
**calculate_score_for_test_data():**
    
    This method will generate the list of top 5 predictions for each sentence combination in the test input.
    Each row in the test input consists of 7 fields; 
    archiveType, proxyObservationType, units, interpretation/variable, interpretation/variableDetail, inferredVariable, inferredVarUnits.
    Since we have 2 chains for prediction, we will split the sentence accordingly.
    1st prediction will be to get the proxyObservationType given the archiveType as a comma-separated sentence.
    2nd prediction will be to get the units and interpretation/variable given the archiveType and proxyObservationType as a comma-separated sentence.
    3rd prediction will be to get the interpretation/variableDetail given archiveType, proxyObservationType, interpretation/variable as a comma-separated sentence
    and so on...
    
    For each sentence that is created, get the accuracy score using the actual value in test input and the list of predictions.
    
    Calculate an average score of predictions for each combination of input sentence.
    
    Depending on previous accuracy calculations we have received accuracy score for Markov Chain predictions = 7.17143
    If the average prediction for a sentence crosses this mark, we can consider Markov Chain to be a good fit for predictions for this archiveType.
    
    Returns:

    None.

Usage
-----

To run the code execute the following command:

.. code-block:: none

   python3 calc_accuracy_mc.py