Model Training for Markov Chains
================================

Markov Chains were inherently developed for predicting the next state in the sequence given the previous state.
Markov Chains have an Inital Probability for the states and a Transition Probability from one state to another.

Functions
---------

**fill_q0(in_dict, dict_type):**
    
    Add initial probabilites for all the items in the dataset to intial probability dict
    eg. items in proxyObservationType, units, interpretation/variable and interpretation/variableDetail
    
    Parameters:
    
    in_dict : dict
        Initial probability dict

    dict_type : dict
        Iterate over this dict to add its values to the initial probability dict.

    Returns:

    None.

**calc_freq_multiple(dataframe_obj, ini_map, \*argv):**
    
    Calculate the frequency of items for all the columns in argv.
    Conditional Probability of last column given all the other columns except the last.

    Parameters:
    
    dataframe_obj : pandas dataframe
        Dataframe object containing training data.

    ini_map : dict
        Contains all the items to be considered for the model.

    \*argv : list
        Contains the names for the columns that are being considered for calculating frequency.

    Returns:
    
    counter_dict : dict
        Containing count for all the items that appear against each item in the last column.

**calc_freq(dataframe_obj, col1, col2, ini_map):**

    Calculate the frequency of items in col2 for each item in column 1.
    Conditional Probability of col2 given column 1

    Parameters:

    dataframe_obj : pandas dataframe
        Dataframe object containing training data.

    col1 : str
        Column for which data is being calculated.

    col2 : str
        Column whose count is being taken.

    ini_map : dict
        Contains all the items to be considered for the model.

    Returns:

    counter_dict : dict
        Containing count for all the items that appear against each item in col1.

**add_extra_keys(all_keys, transition_matrix):**
    
    Add missing items for transition from single key to all items in the dataset.

    Parameters:

    all_keys : set
        Contains all the items that should be in the transition dict for each item.

    transition_matrix : dict
        Transition dict object according to the chain type.

    Returns:

    transition_mat : dict
        Updated dict after adding all the items in the transition dict for each item.

**add_one_smoothing(transition_matrix):**
    
    Add one smoothing to prevent the issue caused due to 0 transition probability from one item to the next.
    Convert counts to log probabilities
    
    Parameters:

    transition_matrix : dict
        Transition dict for all items.

    Returns:

    transition_mat : dict
        Updated transition dict with log probabilities.

Usage
-----
To run the code execute the following command:

.. code-block:: none

   python mctrain.py

Extensions
----------

1. Introduction of new fieldTypes to the sequence

    All the new items for this fieldType will need to be added to the Initial Probability Dict.
    Apart from this the transition from one fieldType to another will change as well.
    Code changes will require to call calc_freq() or calc_freq_multiple() to generate the transition counts for the required columns.
    These transition counts will be added to the main Transition probability dict.
    Calling the add_one_smoothing() method will ensure that there are no 0 probabilities in the Transition Probability Dict
    
