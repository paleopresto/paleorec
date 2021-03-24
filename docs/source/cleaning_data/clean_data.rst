Cleaning the data
=================

This module consists of two subroutines:

**add_to_read_files_list_wiki(root_name, dataset_files_list):**

    This method updates the list of lipd files used for create the training data.
    It first checks if a given lipd file is present in the dataset_files_list
    if yes,then it will create a full path name using the provided root_name,
    else it will use the lipd file available from the wiki.

    Parameters:

    root_name : string
        Root Directory for the files passed in the dataset_files_list
    dataset_files_list : list
        List of files to be read and processed using utils.readLipdFileUtils

    Returns:
    None.

**add_to_read_files_list(root_name, dataset_files_list):**

    This method updates the list of lipd files used for create the training data.
    It adds all the files passed in the dataset_files_list annotated with its complete file path to the read_files_list.

    Parameters:

    root_name : string
        Root Directory for the files passed in the dataset_files_list
    dataset_files_list : list
        List of files to be read and processed using utils.readLipdFileUtils

    Returns:
    None.


**get_data_from_lipd():**

    This passes the read_files_list to the readLipdFileutils which returns a dataframe with proxyObservationType chain and the inferredVariableType chain.

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