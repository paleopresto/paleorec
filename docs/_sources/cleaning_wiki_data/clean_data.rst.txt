Cleaning the data
=================

This module consists of two subroutines:

**get_data_from_lipd():**

    This method creates a list of lipd files to be read to create the training data.
    It first checks if a given lipd file is present in any of the lipdverse datasets, 
    if yes,then it will create a full path name using the root for any of the lipdverse datasets,
    else it will use the lipd file available from the wiki.

    This list of files is then passed to the readLipdFileutils which returns a dataframe with proxyObservationType chain and the inferredVariableType chain.

    Dataframes created:
    
    table_com: pandas dataframe
        Contains information extracted for proxyObservationType.

    inf_table_com: pandas dataframe
        Contains information extracted for inferredVariableType.

    Returns:
    None.


**store_data_as_csv():**
    
    Given the dataframe for proxyObservationType(table_com) and the dataframe with the inferredVariableType(inf_table_com).
    This method merges the two dataframes to create a cleaned dataset for the provided data.

    Returns:
    None.