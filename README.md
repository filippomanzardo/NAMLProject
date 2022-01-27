# Stock Price Prediction using Support Vector Machine

This is the implementation and documentation of the assigned project for _Numerical Analysis for Machine Learning_. As the title suggest, it's based on predicting stock prices exploiting the machine learning model [SVM](data/Cortes-Vapnik1995_Article_Support-vectorNetworks.pdf).
The entire process and theory behind our choices are widely explained in the [report](NAML_Project-Manzardo.pdf).

## Folder Structure
> * [data](data) contains the dataset and results of hypertuning window and shift parameters.
> * [functions](functions) contains Python modules used in the implementation.
> * [models](models) contains sample trained models and images of their structure.
> * [notebook](notebook) contains a Jupiter Notebook example of what has been developed, also the required libraries.
> * [outputs](outputs) has some example output of the experiments.
> * [papers](papers) has some referenced papers.

## Installation

Required Python libraries can be found [here](notebook/requirements.txt), to install them in your system or virtual env:
```
$ pip install -r notebook/requirements.txt
```
To avoid version control errors:
```
$ pip install -r notebook/requirements.txt
```

TA-lib is also needed for calculating technical indicators, if missing, [here](https://github.com/mrjbq7/ta-lib) some references.
