Making predictions
==================

Prediction Logic mainly focuses on processing the input to be according to the training input to be eligible for prediction.
Using the models developed by the training modules, prediction logic returns the next word in the sequence for the fieldType.

The implementation of this module is available through `lipd.net <https://lipd.net/playground>`_ playground. No data is kept on lipd.net. This recommendation is a result of offline training using expert understanding of LiPDverse. 

The sever code for prediction logic and the model files are present on pythonanywhere and hosted on a linux server.

To make any changes to LSTMpredict.py or MCpredict.py file please contact the owners.

Contact : linkedearth@gmail.com

Instructions for use in the flask_app.py server code:

1. If there is a change in data and the model has been retrained on it, then new model files will be created.
2. After changes to LSTMpredict.py please upload the latest model files and the updated LSTMpredict.py file to pythonanywhere. 
3. The flask_app.py code and the predict.py files will pick up the latest model files to work on.

The evalution metrics for the Recommendation System uses the predict logic. 


Routines
--------

.. toctree::
   :maxdepth: 1

   /prediction/LSTMpredict.rst
   /prediction/MCpredict.rst





