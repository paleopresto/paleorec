Model Training for LSTM
=======================

LSTMs are a kind of RNN and function similar to traditional RNNs, its Gating mechanism is what sets it apart.This feature addresses the “short-term memory” problem of RNNs. LSTM’s also has the ability to preserve long-term memory. This is especially important in the majority of Natural Language Processing (NLP) or time-series and sequential tasks.

Functions
---------

**convert_dataframe_to_list(dataframe_obj):**

    Method to return list of all the values in a single row separated by spaces from the dataframe.
    All values that were space separated before are converted to a single word.
    example. Sea Surface Temperature -> SeaSurfaceTemperature

    Parameters:

    dataframe_obj : pandas dataframe
        Dataframe contains the training data.

    Returns:

    new_list : list
        List of input sentences.
    reference_dict : dict
        Mapping of the word to its space-stripped version used for training.

**calculate_unique_chains(dataframe_obj):**

    Method to get unique chains of different lengths from the training data.

    Parameters:

    dataframe_obj : pandas dataframe object
        Data to generate unique chains from.

    Returns:

    None.

**get_data_from_df(lipd_data_df, batch_size, seq_size):**

    Read training data into dataframe for training the model.
    The training data needs to be Label Encoded because LSTM only works with float data.
    Select only num_batches*seq_size*batch_size amount of data to work on.

    Parameters:

    lipd_data_df : pandas dataframe
        Dataframe containing either training sdata.
    batch_size : int
        Used to divide the training data into batches for training.
    seq_size : int
        Defines the sequence size for the training sentences.

    Returns:

    int_to_vocab : dict
        Mapping of the Label Encoding int to text.
    vocab_to_int : dict
        Mapping of the Label Encoding text to int.
    n_vocab : int
        Size of the Label Encoding Dict.
    in_text : list
        Contains the input text for training.
    out_text : list
        Corresponding output for the input text.
    reference_dict : dict
        Mapping of the word to its space-stripped version used for training.

**get_batches(in_text, out_text, batch_size, seq_size):**

    Returns a batch each for the input sequence and the expected output word.

    Parameters:

    in_text : list
        Label Encoded strings of text.

    out_text : list
        Label Encoded Output for each each input sequence.

    batch_size : int
        Parameter to signify the size of each batch.

    seq_size : int
        Parameter to signify length of each sequence. In our case we are considering 2 chains, one of length 3 and the other of length 6.

    Yields:

    list
        batch of input text sequence each of seq_size.

    list
        batch of output text corresponding to each input.

**get_loss_and_train_op(net, lr=0.001):**

    We are using CrossEntropy as a Loss Function for this RNN Model since this is a Multi-class classification kind of problem.

    Parameters:

    net : neural network instance
        Loss function is set for the Neural Network.

    lr : float, optional
        Defines the learning rate for the neural network. The default is 0.001.

    Returns:

    criterion : Loss function instance
        Loss Function instance for the neural network.

    optimizer : Optimizing function instance
        Optimizer used for the neural network.

**print_save_loss_curve(loss_value_list, chain):**

    Method to save the plot for the training loss curve.

    Parameters:

    loss_value_list : list
        List with the training loss values.

    chain : str
        To differentiate between the proxyObservationTypeUnits chain from the proxyObservationType & interpretation/variable chain.

    Returns:

    None.

**train_RNN(int_to_vocab, vocab_to_int, n_vocab, in_text, out_text, seq_size, for_units = False):**

    Method to train an lstm model on in_text and out_text.
    This method will save the model for the last epoch.

    Parameters:

    int_to_vocab : dict
        Mapping of the Label Encoding int to text.

    vocab_to_int : dict
        Mapping of the Label Encoding text to int.

    n_vocab : int
        Size of the Label Encoding Dict.

    in_text : list
        Contains the input text for training.

    out_text : list
        Corresponding output for the input text.

    for_units : boolean, optional
        Flag to signify if model is training for the chain archiveType -> proxyObservationType -> units. The default is False.

    Returns:

    None.

Usage
-----
1. Please change the directory to /training/lstm/
2. The commandline takes as input 2 arguments '-e' for the number of epochs we want to train the model and '-l' the learning rate for the Recurrent Neural Network.
3. To understand the training loss, this module also generates a loss curve. Depending on where the training file is executed from i.e. from jupyter notebook or commandline, the file will be saved or displayed on the GUI.

To run the code execute the following command:

.. code-block:: none

    cd /training/lstm/
    python train_lstm.py -e 150 -l 0.01
    python train_lstm.py -e 100 -l 0.001 -u (For Units)

1. Alternatively, to execute from the jupyter notebook:

   a. Navigate to the **training** folder.
   b. Within that open the **lstm** folder.
   c. Click on the **run_train_lstm.ipynb**.
   d. You can scroll down past to the end and run the latest commands in the last 2 cells.
   e. Going over the output of the other cells will show the training loss for other epochs and learning rates.

There is an existing `binder <https://mybinder.org/v2/gh/paleopresto/paleorec/HEAD>`_., just remember to commit the data to GitHub before launching.

Extensions
----------

1. Introduction of new fieldTypes to the sequence

    The only changes will be to the flags.seq_size field to indicate the new sequence size.
    The model will now be trained on the new sentence length.

Check your work: Learning curves
--------------------------------

The model files created after training are stored at data/model_lstm:
* model_lstm_interp_timestamp.pth
* model_lstm_units_timestamp.pth
* model_token_info_timestamp.txt
* model_token_units_info_timestamp.txt

If you did not perform this step in a Jupyter Notebook, the learning curves were saved at data/loss:
* proxy_interp_training_loss_e_100_l_0.01_timestamp.png
* proxy_units_training_loss_e_100_l_0.01_timestamp.png

where `timestamp` takes the form: mmddyyyy_hhmmss (e.g., 05252021_143300 for a file created on May 25th 2021 at 2:33pm).
