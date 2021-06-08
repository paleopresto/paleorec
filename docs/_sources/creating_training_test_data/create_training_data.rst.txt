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

**discard_less_frequent_values_from_data():**
    
    This method reduces the subset of data to the fields in the chain, 
    i.e archiveType, proxyObservationType, units, interpretation/variable, interpretation/variableDetail, inferredVariable, inferredVarUnits.
    There are various tasks perfomed in this function.
    
    Create a dict to store autocomplete information for each fieldType.
    
    Generate a counter for the values in each column to understand the distribution of each individual field.
    Manually decide whether to eliminate any co 1 values within each field.
    Uncomment the code to print each of the counter fields to make the decision.
    
    Update the dataframe by discarding those values from there as well.

    Returns:
    None.

**downsample_archives_create_final_train_test_data():**
    
    Manually decide based on the counter for archiveTypes which archiveTypes need to be downsampled.
    Currently we are downsampling Wood and Marine Sediment to include 350 samples of each.
    We are including all samples for all the other archiveTypes.
    
    Simulataneously creating a test dataset by resampling from the training data.
    Since we do not even distribution of data across each class, we have used 'stratify' during resample.
    This will help us even out the distribution of data across all classess in the provided dataset.
    
    Returns:
    None.