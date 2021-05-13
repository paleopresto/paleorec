Creating Training and Test Data
===============================

This module consists of following subroutines:

**read_latest_data_for_training():**

    Method to read the latest file cleaned using utilities in cleaning_wiki_data/clean_data.py
    The latest data is picked up using the utilities file which uses os.ctime
    Reads the csv and stores the data into the common_lipdverse_df dataframe.
    
    Returns:
    None.   

**manually_clean_data_by_replacing_incorrect_values():**
    
    Manual task to replace the following data in the dataframe with its alternative text.
    Could not eliminate these errors while reading lipd files using code in cleaning_wiki_files/clean_data.py
    Replace the data in place within the dataframe.
    
    Returns:
    None.

**write_autocomplete_data_file():**
    
    Writes the data to autocomplete_file used for autocomplete suggestions on the UI

    Returns:
    None.

**take_user_input():**

    Method to take user input to eliminate any-co-k occurance of data.
    This method validates the input to be a positive integer and returns it.

    Returns:

    int
        Value for any-co-k elimination.

**editDistDP(str1, str2, m, n):**

    Calculates the edit distance between str1 and str2.

    Parameters:

    str1 : string
        Input string 1.
    str2 : TYPE
        Input string 2.
    m : int
        len of string 1
    n : int
        len of string 2

    Returns:

    int
        Edit distance value between str1 and str2.

**discard_less_frequent_values_from_data():**
    
    This method reduces the subset of data to the fields in the chain, 
    i.e archiveType, proxyObservationType, units, interpretation/variable, interpretation/variableDetail, inferredVariable, inferredVarUnits.

    Create a dict to store autocomplete information for each fieldType.

    Generate a counter for the values in each column to understand the distribution of each individual field.
    Manually decide whether to eliminate any co 1 values within each field by taking user input.

    Update the dataframe by discarding values from there.

    Returns:
    None.

**downsample_archive(archiveType, downsample_val):**
    
    Method to downsample an archiveType to the provided value in the params.
    This module also generates the test data for the given archiveType.

    Parameters:

    archiveType : str
        Archive Type to downsample.

    downsample_val : int
        Number of samples the archiveType needs to be reduced to.

    Returns:
    None.

**get_label_set_for_input(dataframe_obj, col1, col2):**

    Calculate the get items in col2 for each item in column 1.

    Parameters:

    dataframe_obj : pandas dataframe
        Dataframe object containing training data.
    col1 : str
        Column for which data is being calculated.
    col2 : str
        Column whose item is being taken.

    Returns:

    counter_dict : dict
        Containing set of all the items that appear against each item in col1.

**update_ground_truth_dict(temp_dict):**

    Method to add values from one dict to another. 
    If a key is present append the list of values to the already created list.

    Parameters:

    temp_dict : dict
        Dict whose values need to be added to the ground truth dict.

    Returns:

    None.

**generate_ground_truth_label_info(final_df_test):**

    Method to collect the list of all possible next values for a given field.
    Example:
        Given Marine Sediment
        Ouput for Proxy Observation Type  = ["Notes", "Mg/Ca", "Bsi", "Caco3", "Uk37", "Mgca", "IRD", "D18O", "37:2Alkenoneconcentration", "TOC", "D18O.Error", "DBD", "D13C", "Dd", "D13C.Error", "Foram.Abundance"]

    Parameters:

    final_df_test : pandas dataframe
        Final Dataframe on which Counts for the various fields are calculated.

    Returns:

    None.

**calculate_counter_info(final_df):**

    Method to get list of all possible values for each fields in the recommendation system.

    Parameters:

    final_df : pandas dataframe
        Pandas dataframe consisting of training information.

    Returns

    None.

**downsample_archive(archiveType, downsample_val):**

    Method to downsample an archiveType to the provided value in the params.
    This module also generates the test data for the given archiveType.

    Parameters:

    archiveType : str
        Archive Type to downsample.
    downsample_val : int
        Number of samples the archiveType needs to be reduced to.

    Returns:

    None.

**downsample_archives_create_final_train_test_data():**
    
    Manually decide based on the counter for archiveTypes which archiveTypes need to be downsampled.
        
    Two approaches were tried for creating the test data from the generated data.

    First was creating a test dataset by resampling from the training data.
    Since we do not even distribution of data across each class, we have used 'stratify' during resample.
    This will help us even out the distribution of data across all classess in the provided dataset.

    Second approach is to keep aside 20% of the generated data as unseen test data, while 80% of the data would be used as the training data.
    Using a bar plot distribution tried to verify the ratio of the archives to proxyObservationType in the training and test data are nearly equal.

    After the final training data is procured, the ground truth data file is created which is used in the final prediction.

    
    Returns:
    None.

