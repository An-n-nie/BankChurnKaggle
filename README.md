![](UTA-DataScience-Logo.png)

# Bank Churn Kaggle Challenge

* This repository holds an attempt to apply Histogram-based Gradient Boosting to predict churn rates at a bank provided through (["Binary Classification with a Bank Churn Dataset"](https://www.kaggle.com/competitions/playground-series-s4e1/overview) Kaggle challenge. 

## Overview

* This section could contain a short paragraph which include the following:
  * **Definition of the tasks / challenge**  Ex: The task, as defined by the Kaggle challenge is to use a time series of 12 features, sampled daily for 1 month, to predict the next day's price of a stock.
  * **Your approach** Ex: The approach in this repository formulates the problem as regression task, using deep recurrent neural networks as the model with the full time series of features as input. We compared the performance of 3 different network architectures.
  * **Summary of the performance achieved** Ex: Our best model was able to predict the next day stock price within 23%, 90% of the time. At the time of writing, the best performance on Kaggle of this metric is 18%.

## Summary of Workdone

### Data

* Data:
  * Type: Tabular
    * Input: medical images (1000x1000 pixel jpegs), CSV file: image filename -> diagnosis
    * Input: CSV file of features, output: signal/background flag in 1st column.
  * Size: 165034 rows, 14 columns
  * Instances (Train, Test, Validation Split): how many data points? Ex: 1000 patients for training, 200 for testing, none for validation

#### Preprocessing / Clean up

* Describe any manipulations you performed to the data.

#### Data Visualization

Show a few visualization of the data and say a few words about what y

### Problem Formulation

* Define:
  * Input / Output
  * Models
    * Describe the different models you tried and why.
  * Loss, Optimizer, other Hyperparameters.

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* HistGradientBoost have a slight edge over Decision Tree in this dataset. However, HistGradientBoost, DecisionTree, and RandomForest all performed in similar ranges for this problem. If aiming for the sake of simplicity, Decision Tree can do well without withholding too much potentials. 

### Future Work

* Other machine learning algorithms like XGBoost, CatBoost, and LightGMBoost are promissery steps for the future. Many others have tried these algortihms and they proved, in general, to be very effective at solving and scoring high marks in Kaggle challenges.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* The repository should be read from Data Load n Initial Look, Data Viz, Data Clean N Prep, ML, and Bank Churn Final to obtain better understanding of the workflow. 
  * Data_Load_N_Initial_Look.ipynb: 
  * Data_Viz.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * Data_Clean_N_Prep.ipynb: Creates various visualizations of the data.
  * ML.ipynb: Contains functions that build the various models.
  * Bank_Churn_Final.ipynb: Contains all the preprocessing and machine learning processes necessary to obtain results and convert them into the proper submission format.
  * submission.csv: Submission file created after all of the work is done. 
  * sample_submission.csv: A sample of the submission format provided by the challenge.
  * train.csv: Train dataset provided by the challenge.
  * test.csv: Test dataset provided by the challenge. 

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Kaggle Dataset: Walter Reade and Ashley Chow. Binary Classification with a Bank Churn Dataset . ([https://kaggle.com/competitions/playground-series-s4e1](https://kaggle.com/competitions/playground-series-s4e1)), 2024. Kaggle.







