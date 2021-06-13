Prediction using LSTM
=====================

This module contains the code necessary to make a recommendation based on a LSTM.

Functions
---------

**get_latest_file_with_path(path, \*paths):**
    
    Method to get the full path name for the latest file for the input parameter in paths.
    This method uses the os.path.getctime function to get the most recently created file that matches the filename pattern in the provided path. 

    Parameters:

    path : string
        Root pathname for the files.

    \*paths : string list
        These are the var args field, the optional set of strings to denote the full path to the file names.

    Returns:

    latest_file : string
        Full path name for the latest file provided in the paths parameter.

Class
-----
LSTMpredict
^^^^^^^^^^^

Functions
"""""""""

**predict(self, device, net, words, vocab_to_int, int_to_vocab, names_set):**
    
    Returns the list of top 5 predictions for the provided list of words using the model stored in net.
    The device is initialized to CPU for the purpose of the predictions.

    Parameters:

    device : torch.device
        Device type to signify 'cpu' or 'gpu'.

    net : torch.module
        Instance of LSTM created using RNN Module.

    words : list
        List of strings used for predicting the next string in the list of words.

    vocab_to_int : dict
        Mapping of strings to int used to embed the input strings.

    int_to_vocab : dict
        Mapping of int to string used in the process of running the model predictions.

    names_set : dict
        Mapping of fieldType(example proxyObsType, interpretation/variable) and list of all the possible values the field can take.

    Returns:

    list
        Top 5 recommendations for the next string in the sequence of words.

**predictForSentence(self, sentence, isInferred = False):**
    
    This method is used from the Flask Server Code API. 
    This method handles the initialization of the lstm model for the two different prediction chains we are using in our system.
    
    archiveType -> proxyObservationType -> units
    archiveType -> proxyObservationType -> interpretation/variable -> interpretation/variableDetial -> inferredVariable -> inferredVarUnits
    
    Depending on the length of the input sentence and the variableType, it chooses the output that will be returned to the server.
    
    If the variableType == measured
    then we will be considering the complete chain for prediction
    example: If sentence length = 1, it contains the archiveType and output = prediction for proxyObservationType
    example: If sentence length = 2, it contains the archiveType and proxyObservationType and output = units and interpretation/variable
    so on..
    
    If the variableType == inferred
    then we will be considering the top attributes in the chain from proxyObservationType to interpretation/variableDetail 
    to predict the top 5 inferredVariable as the output .
    
    Parameters:

    sentence : string
        Input sentence to predict the next field.

    isInferred : boolean, optional
        True if variableType == 'inferred'. The default is False.

    Returns:

    dict
        Contains the result list of predictions as the value.
        Depending on the length of the input sentence and the variableType,
        the dict can contain one item corresponding to key '0' or two items corresponding to the two keys '0' and '1'.

Usage
-----

LSTMpredict.py module is used for accuracy calculation in the /accuracy_calc/lstm directory. For more information check out the documentation for Metrics Calculation.

.. toctree::
    :maxdepth: 1

    /../accuracy_calculation/fang_metrics.rst
    
