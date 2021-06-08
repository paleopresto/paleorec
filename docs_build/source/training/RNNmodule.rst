RNN module
==========

Defines a nn.Module from pytorch where we define the architecture for the Recurrent Neural Network used for the Recommedation System.

The archicture consititutes of Embedding Layer of size 64 * size of input words. This is followed by an LSTM layer of 64(embedding size) * 64(lstm size).
Finally a Dense Linear Layer is used to give the output for the model.

.. code-block:: none

    RNNModule(
        (embedding): Embedding(91, 64)
        (lstm): LSTM(64, 64, batch_first=True)
        (dense): Linear(in_features=64, out_features=91, bias=True)
    )

