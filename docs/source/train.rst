Training the Models
===================

We have modeled our use case as follows:
   
   Given a comma-separated input sequence, give next word for the fieldtype. 
   The two sequences we are currently using are:
   
   archiveType -> proxyObservationType -> proxyObservationTypeUnits
   archiveType -> proxyObservationType -> interpretation/variable -> interpretation/variableDetail -> inferredVariable -> inferredVarUnits

We are implementing Sequential Recommendation System to annotate paleoclimate data. Since the data we are working with is in textual format, this can be viewed as a text generation problem. Traditionally frequent pattern mining, k-nearest neighbors, markov chains were used for this kind of problem. 

We have implemented a combination of First-order Markov Chains and Higher-order Markov Chains.
The code is available at `markov_chain in <https://github.com/paleopresto/recommender/tree/main/paleorec_research>`_.

In the recent times Recurrent Neural Networks have shown immense performance gains. Gated Recurrent Units(GRUs) are a type of RNN which effectively combat the fading/exploding gradient problem. Long Short Term Memory(LSTMs) are another type of RNN which are similar to GRUs, built of a different architecture but achieving the same results. We have chosen LSTM over GRUs for performance reasons.


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

Problems
--------

1. On adding more datasets and training the 2 models, Markov Chain models do not scale with increase in data. The model file created grows exponentially and demands immense prediction power. 
2. Using Glove Embeddings did not prove to be helpful to the formulation of the LSTM chain, since we do not have word embeddings in our case. Instead Label Encoding the data proves to be an effective solution.