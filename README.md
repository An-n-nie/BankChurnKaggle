![](UTA-DataScience-Logo.png)

# Bank Churn Kaggle Challenge

* This repository holds an attempt to apply Histogram-based Gradient Boosting to predict churn rates at a bank provided through (["Binary Classification with a Bank Churn Dataset"](https://www.kaggle.com/competitions/playground-series-s4e1/overview) Kaggle challenge. 

## Overview

* The goal of the Kaggle challenge was to use a tabular dataset containing various banking information to predict whether a customer keeps their account or churn it. This repository considers the task at hand a binary classification problem and uses different machine learning models to sample their performances. Then, the model with the best performance will undergo some model tuning before proceeding to apply it to the final dataset. In this case, Histogram-based Gradient Boosting was the best performing model, achieving an average area under the ROC curve score of 88.75%. The last-updated score on the Kaggle challenge's leader board was 90.59%.

## Summary of Workdone

### Data

* Data:
  * Type: Tabular
    * Input: CSV files (train, test) of different bank-related features (tenure, credit score, balance, etc.)
    * Output: signal/background rates
  * Size:
    * Train: 165034 rows, 14 columns (last column - "Exited" rates for signal/background)
    * Test: 110023 rows, 13 columns
  * Instances: Train.csv file is split into
    * 60% Train
    * 20% Validation
    * 20% Test


#### Preprocessing / Clean up

* Null: No null values found
* Duplicates: some were found after removing id and customer id columns, revealing customers with same surnames and same account balances and estimated salaries. These were assumed to be duplicates and removed. 
* Categorical columns: Geography and Gender columns were onehot encoded using OneHotEncoder from Scikitlearn.
* Numerical columns:
  * Credit score, balance, age, tenure, estimated salary and number of products columns were scaled using StandardScaler from Scikitlearn. (both MinMaxScaler and StandardScaler were used and both gave same results.)
  * The rest of the numerical columns were already in good range (i.e. binary)


#### Data Visualization
* There were some class imbalance found between the signal and background variables.
* Out of all of the numerical features, Age seem to be a good separator for the siggnal and bakcground rates. 


### Problem Formulation

* Define:
  * Input / Output
  * Models
    * Decision Tree:
    * Histogram-based Gradient Boosting:
    * Random Forest:
    * Logistic Regression:
    * K-Nearest Neighbors:
  * Parameters:
    * All models were 

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Models are evaluated using the area under the ROC curve
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

* The repository should be read from Data Load n Initial Look, Data Viz, Data Clean N Prep, ML, and Bank Churn Final to obtain a good understanding of the workflow. 
  * Data_Load_N_Initial_Look.ipynb: 
  * Data_Viz.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * Data_Clean_N_Prep.ipynb: Creates various visualizations of the data.
  * ML.ipynb: Contains functions that build the various models.
  * Bank_Churn_Final.ipynb: Contains all the preprocessing and machine learning processes necessary to obtain results and convert them into the proper submission format.
  * submission.csv: Submission file created after all of the work is done. 
  * sample_submission.csv: A sample of the submission format provided by the challenge.
  * train.csv: Train dataset provided by the challenge.
  * test.csv: Test dataset provided by the challenge. 


### Software Setup
* Libraries like pandas, matplotlib, numpy, math, and scipy are needed and can be called using import. 
* Scikit learn needs to be downloaded into terminal using pip install before importing it into notebook.


### Data

* Data for this challenge can be downloaded through the (["Kaggle Challenge website"](https://www.kaggle.com/competitions/playground-series-s4e1/overview). Most convenitently, the API for Kaggle can be downloaded into Terminal which should then be relocated to the correct .kaggle folder. Then, use kaggle command to download datasets from the kaggle websites into jupyter notebook or any preferred environment. 

### Training

* Different models are initiated, fitted and trained on the train portion (60% of total data points) of train.csv dataset.

#### Performance Evaluation

* Performance of the models can be evaluated via calculating the area under the ROC curve scores on the validation and test portions of the dataset. Cross-validation tests should also be run to ensure unbiased results. 


## Citations

* Kaggle Dataset: Walter Reade and Ashley Chow. Binary Classification with a Bank Churn Dataset . ([https://kaggle.com/competitions/playground-series-s4e1](https://kaggle.com/competitions/playground-series-s4e1)), 2024. Kaggle.







