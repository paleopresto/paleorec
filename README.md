# Paleorec [![](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/paleopresto/paleorec/blob/main/LICENSE) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5108095.svg)](https://doi.org/10.5281/zenodo.5108095)

**A Recommender system for the annotation of paleoclimate data**

The <a href="https://lipd.net/playground" target="_blank">lipd.net playground</a> enables paleoclimate scientists to create and edit files in the [LiPD](https://lipd.net) format. PaleoRec is a recommender system that assists these scientists in the recommendation task. 

This repository contains the code to train the deep learning at the heart of PaleoRec. We have modeled the recommendation process as a *Sequential Recommendation System* problem identifying 2 main sequence chains in the data : <br>
1. archiveType -> proxyObservationType -> units
2. archiveType -> proxyObservationType -> interpretation/variable -> interpretation/VariableDetail -> inferredVariable -> inferredVariableUnits

## Demonstration

For a quick Demo, please run the cells in the demo\LSTM_Demo.ipynb by launching the binder in the browser. Note that the first time you launch the binder, it may take some time to build.<br>

**Things to remember while running the notebook:**<br>
1. After you launch the binder, navigate to the demo folder.
2. Inside the demo folder choose LSTM_Demo.
3. In the File Menubar at the top, click on Cell -> Run All
4. Using the widgets simulate the selection process.
5. To change any value from variableType onwards, please repeat from step3.
<br>

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/paleopresto/paleorec/HEAD)

## Deployment

PaleoRec is currently deployed on the [LiPD playgroud](https://lipd.net/playground). 

## Research

For more details about PaleoRec, including its performance, see [this repository](https://github.com/paleopresto/recommender). 
