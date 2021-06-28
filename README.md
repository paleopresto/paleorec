# Paleorec [![](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/paleopresto/paleorec/blob/main/LICENSE)

## Recommender system for the annotation of paleoclimate data

The <a href="https://lipd.net/playground" target="_blank">lipd.net playground</a> enables paleoclimate scientists to create and edit files in the [LiPD](https://lipd.net) format. PaleoRec is a recommender system that assists these scientists in the recommendation task. 

This is a repository for providing offline recommendation to the users to provide intuitive recommendation for various important fields in Paleo Measurement Table. 

**We have 2 models** <br>
1. Markov Chains (Accuracy Score - 7.0857)
2. Deep NN - LSTM (Accuracy Score - 768571)

We have modeled the recommendation process as a *Sequential Recommendation System* problem identifying 2 main sequence chains in the data : <br>
1. archiveType -> proxyObservationType -> units
2. archiveType -> proxyObservationType -> interpretation/variable -> interpretation/VariableDetail -> inferredVariable -> inferredVariableUnits

For a quick Demo, please run the cells in the demo\MC_Demo.ipynb or demo\LSTM_Demo.ipynb by launching the binder in the browser.<br>

**Things to remember while running the notebook:**<br>
1. After you launch the binder, navigate to the demo folder.
2. Inside the demo folder choose LSTM_Demo.
3. In the File Menubar at the top, click on Cell -> Run All
4. Using the widgets simulate the selection process.
5. To change any value from variableType onwards, please repeat from step3.
<br>

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/paleopresto/paleorec/HEAD)
