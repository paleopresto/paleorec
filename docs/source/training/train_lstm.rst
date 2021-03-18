Model Training for LSTM
=======================

LSTMs are a kind of RNN and function similarly to traditional RNNs, its Gating mechanism is what sets it apart.This feature addresses the “short-term memory” problem of RNNs. LSTM’s also has the ability to preserve long-term memory. This is especially important in the majority of Natural Language Processing (NLP) or time-series and sequential tasks.

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

**get_data_from_file(train_file, batch_size, seq_size, for_units = False):**
    
    Read training data into dataframe for training the model.
    The training data needs to be Label Encoded because LSTM only works with float data.
    Select only num_batches x seq_size x batch_size amount of data to work on.

    Parameters:

    train_file : string
        File path for the training data.

    batch_size : int
        Used to divide the training data into batches for training.

    seq_size : int
        Defines the sequence size for the training sentences.

    for_units : boolean, optional
        Flag to signify if model is training for the chain archiveType -> proxyObservationType -> units. The default is False.

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
To run the code execute the following command:

.. code-block:: none

   python train_lstm.py

Extensions
----------

1. Introduction of new fieldTypes to the sequence

    The only changes will be to the flags.seq_size field to indicate the new sequence size.
    The model will be now trained on the new sentence length.

