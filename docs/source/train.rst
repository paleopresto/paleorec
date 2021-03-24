Training the Models
===================

We have modeled our use case as follows:
   
   Given an comma-separated input sequence, give next word for the fieldtype. 
   The two sequences we are currently using are:
   
   archiveType -> proxyObservationType -> proxyObservationTypeUnits
   archiveType -> proxyObservationType -> interpretation/variable -> interpretation/variableDetail -> inferredVariable -> inferredVarUnits

Since this can be viewed as a text generation problem, we have implemented the solution as a Sequential Recommendation System Problem.
The two most suitable candidates that have proven to be effective are Markov Chains and LSTM.
We have implemented both the algorithms for our use case.

Routines
--------

.. toctree::
   :maxdepth: 1

   /training/train_lstm.rst
   /training/RNNmodule.rst
   /training/mctrain.rst
   
Extensions
----------

There are 2 possible extensions to the problem:
1. Introduction of new fieldTypes to the sequence; addressed for each model
2. Introducing a new model for training

   The model will receive as input a comma-separated sequence of the chain. Given the input sequence, it should predict the next value for the next field type in the sequence.
   If any neural network is used, it will require Label Encoding of the input to train the model.
   Using Glove Embeddings did not prove to be helpful to the formulation of the LSTM chain, since we do not have word embeddings in our case.

