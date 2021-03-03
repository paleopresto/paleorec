# Paleorec
## Recommender system for paleoclimate data

The <a href="https://lipd.net/playground" target="_blank">lipd.net playground</a> enables climate scientists and users of the lipd data format to create, edit and upload lipd files.
This is a repository for providing offline recommendation to the users to provide intuitive recommendation for various important fields in Paleo Measurement Table. 

We have 2 models <br>
1. Markov Chains (Accuracy Score - 7.0857)
2. Deep NN - LSTM (Accuracy Score - 768571)

We have modeled the recommendation process as a Sequential Recommendation System problem.<br>
Identifying 2 main sequence chains in the data : <br>
1. archiveType -> proxyObservationType -> units
2. archiveType -> proxyObservationType -> interpretation/variable -> interpretation/VariableDetail -> inferredVariable -> inferredVariableUnits

The repository structure is as follows:

```

│   environment.yml
│   LICENSE
│   README.md
│   requirements.txt
│   setup.cfg
│   setup.py
│
├───accuracy_calculation
│   ├───lstm
│   │       calc_accuracy_lstm.py
│   │
│   └───markovchain
│           calc_accuracy_mc.py
│
├───cleaning_wiki_data
│       clean_data.py
│
├───creating_train_test_data
│       create_training_data.py
│
├───data
│   ├───csv
│   │       common_lipdverse_inferred.csv
│   │       common_lipdverse_inferred_20210223_233830.csv
│   │       common_lipdverse_table.csv
│   │       common_lipdverse_table_20210223_233830.csv
│   │       lipdverse_downsampled.csv
│   │       lipdverse_downsampled_20210222_125233.csv
│   │       lipdverse_test.csv
│   │       lipdverse_test_20210222_125233.csv
│   │       merged_common_lipdverse_inferred.csv
│   │       merged_common_lipdverse_inferred_20210223_233830.csv
│   │       wood_inferred_data.csv
│   │
│   ├───model_lstm
│   │       model_lstm_interp_20210222_152046.pth
│   │       model_lstm_units_20210222_152051.pth
│   │       model_token_info_20210222_152037.txt
│   │
│   ├───model_mc
│   │       model.txt
│   │       model_mc_20210222_150245.txt
│   │
│   └───wiki_lipd_files (705 lipd files)
│
├───demo
│       LSTM_Demo.ipynb
│       MC_Demo.ipynb
│
├───example
│       output_archive_proxy_intVar_intVarDet.txt
│       output_archive_proxy_units.txt
│       Sample_usage_for_LSTM.txt
│       Sample_usage_for_MC.txt
│       test_archive_1.txt
│
├───prediction
│   ├───lstm
│   │       LSTMpredict.py
│   │
│   └───markovchain
│           MCpredict.py
|
├───training
│   ├───lstm
│   │       RNNModule.py
│   │       train_lstm.py
│   │   
│   │
│   └───markovchain
│           mctrain.py
│
├───utils
       fileutils.py
       inferredVarTypeutils.py
       proxyObsTypeutils.py
       readLipdFileutils.py
   
```

1. accuracy_calculation : Formulated an accuracy measure for the sequential prediction model. 
2. cleaning_wiki_data : Currently the data considered for training is a combination from wiki.linked_earth.com and lipdverse.org.<br>
    Cleaning involved checking for spelling errors, repeated values like Temperature1, Temperature2, and mapping all the values to a base form, eg d18O -> D18O, D180 -> D18O<br>
    <br>
    Please download the required datasets from lipdverse.org. <br>
    Usage: python3 clean_data.py -p [path-to-PAGES2k-dataset] -t [path-to-Temp12k-dataset] -i [path-to-iso2k-dataset] -pm [path-to-Palmod-dataset] <br>
    All the commandline arguments with '-' are optional.<br>

3. creating_train_test_data : Training and Test data is a downsampled version of the entire dataset to prevent any class-imbalance problems
4. data<br>
    a. csv : Consists of files with training and test data.<br>
    b. wiki_lipd_files : Since training data is created using the wiki, uploaded the files as a backup for generating training data.<br>
5. utils: 
    a. fileutils : code to resolve the most recent file with the filename pattern (example common_lipdverse_inferred_20210223_233830.csv reduced to common_lipdverse_inferred_*.csv)<br>
    b. inferredVarTypeutils : code to resolve the given string to the most probable value of inferredVariableType using a mapping from the previous data<br>
    c. proxyObsTypeutils : code to resolve the given string to the most probable value of proxyObservationType using a mapping from the previous data<br>
    d. readLipdFileutils : code to read a list of lipd files and convert the data to pandas dataframe



For a quick Demo, please run the cells in the demo\MC_Demo.ipynb or demo\LSTM_Demo.ipynb by launching the binder in the browser.<br>
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/paleopresto/paleorec/HEAD)