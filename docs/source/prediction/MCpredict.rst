Prediction using Markov Chain 
=============================

This module contains the code necessary to make a recommendation based on a Markov Chain.

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
MCpredict
^^^^^^^^^

Functions
"""""""""

**get_inner_list(self, in_list):**
    
    Backtracking code to recursively obtain the item name from the hierachial output list.

    Parameters:

    in_list : list/ tuple
        Either a list object or tuple whose data is retreived.
        
    Returns:

    list
        Condensed hierarchial version of the list without probabilities.

**pretty_output(self, output_list):**
    
    Get the item list without the probabilities.

    Parameters:

    output_list : list
        Output List after complete processing..

    Returns:

    out_dict : dict
        Ordered Dict with level as the key and value as the condensed list for each level.
        
    Example:

        input: [[(-1.0286697494934511, 'Wood')], [(-1.8312012012793524, 'Trsgi')], 
                [[(-2.5411555001556785, 'NA'), (-6.618692944061398, 'Wood'), (-6.618692944061398, 'MXD'), (-6.618692944061398, 'LakeSediment'), (-6.618692944061398, 'Composite')]]]

        output: {'0': ['Wood'], '1': ['Trsgi'], '2': ['NA', 'Wood', 'MXD', 'LakeSediment', 'Composite']}


**get_max_prob(self, temp_names_set, trans_dict_for_word, prob):**

    Find the maximimum items from a list stream using heapq.
    We will only pick those items that belong to the category we are interested in.
    Example : only recommend values in Units for Units.

    Parameters:

    temp_names_set : set
        Set containing the items in the category.

    trans_dict_for_word : dict
        Transition probability dict for the start word.

    prob : float
        The probability of the start word.

    Returns:

    list
        Contains the top 5 recommendation for the start word.

**back_track(self, data, name_list_ind, sentence = None):**

    Function to get top 5 items for each item in sequence

    Parameters:

    data : list/str
        Input sequence.

    name_list_ind: int
        Index for names_list dict. 
        Used to predict only proxyObservationType after Archive, 
        and not give recommendations from other category.

    Returns:

    list
        Output list for the input sequence.

**get_ini_prob(self, sentence):**

    Method to find the transition probability for the given sentence.
    For the first word we use the initial probability and for the rest of the sentence we use the transition probability for getting the next word.

    Parameters:

    sentence : str
        Input string sequence for which we have to predict the next sequence.

    Returns:

    output_list : list
        Output list containing the probability and word for each stage of the sequence.

    sentence : list
        Sentence strip and split on space and returned for further use.

**predict_seq(self, sentence, isInferred = False):**

    Predict the top 5 elements at each stage for every item in the chain
    There are 2 chain types:

        archive -> proxyObservationType -> units, 

        archive -> proxyObservationType -> interpretation/variable, interpretation/variableDetail
        ->inferredVariable -> inferredVarUnits
    
    We do not include inferredVariableType and inferredVarUnits in the sequential prediction, 
    but provide the recommendation after the interpretation/variableDetail has been selected.
    
    If isInferred == True, then we will choose the top value in prediction for the chain given the archiveType
    example:

        archiveType = MarineSediment

        proxy = D180

        interpretation/variable = NA

        interpretation/variableDetail = NA

    then based on this generate the top 5 predictions for inferredVariable
    
    
    Parameters:

    sentence : str
        Input sequence.

    Returns:

    output_list : dict
        Dict in hierarchial fashion containing top 5 predictions for value at each level.
    
    Example:

    input: 'Wood'
    intermediate output: 
                
                [[(-1.0286697494934511, 'Wood')],
                [[(-2.8598709507728035, 'Trsgi'), (-3.519116579657067, 'ARS'), (-3.588109451144019, 'EPS'), (-3.701438136451022, 'SD'), (-3.701438136451022, 'Core')]], 
                [[
                [(-3.5698252496491296, 'NA'), (-7.647362693554849, 'Wood'), (-7.647362693554849, 'MXD'), (-7.647362693554849, 'LakeSediment'), (-7.647362693554849, 'Composite')], 
                [(-4.628778704511761, 'NA'), (-8.029976086173917, 'Wood'), (-8.029976086173917, 'MXD'), (-8.029976086173917, 'LakeSediment'), (-8.029976086173917, 'Composite')], 
                [(-4.744541310700955, 'NA'), (-8.076745820876159, 'Wood'), (-8.076745820876159, 'MXD'), (-8.076745820876159, 'LakeSediment'), (-8.076745820876159, 'Composite')], 
                [(-4.936909607836329, 'NA'), (-8.15578543270453, 'Wood'), (-8.15578543270453, 'MXD'), (-8.15578543270453, 'LakeSediment'), (-8.15578543270453, 'Composite')], 
                [(-4.971198681314961, 'NA'), (-6.803780145063271, 'NotApplicable'), (-8.190074506183162, 'Wood'), (-8.190074506183162, 'MXD'), (-8.190074506183162, 'Composite')]
                ]]]

    final output: {'0': ['Trsgi', 'ARS', 'EPS', 'SD', 'Core']}
    
Usage
-----

MCpredict.py module is used for accuracy calculation in the /accuracy_calc/markovchain directory. For more information check out the documentation for Accuracy Calculation.

.. toctree::
    :maxdepth: 1

    /../accuracy_calculation/calc_accuracy_mc.rst