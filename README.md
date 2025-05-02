# Bank Churn Kaggle Challenge

This repository explores the use of Histogram-based Gradient Boosting to predict customer churn in a bank as part of the [Binary Classification with a Bank Churn Dataset](https://www.kaggle.com/competitions/playground-series-s4e1/overview) Kaggle challenge. 

## Overview

The goal of the Kaggle challenge was to use a tabular dataset containing various banking information to predict whether a customer etains thier account or chooses to close it (churn). This repository considers the task at hand a binary classification problem and evaluates the performance of several machine learning models. Then, the model with the best performance is further fine tuned before being applied to the final dataset. In this case, Histogram-based Gradient Boosting was the best performing model, achieving an average area under the ROC curve score of 88.75%. The last-updated score on the Kaggle challenge's leader board was 90.59%.

## Summary of Workdone

### Data

**Type**: Tabular data (CSV format)  
- **Input**: Banking features such as tenure, credit score, balance, etc.  
- **Output**: Binary target column ("Exited"), where 0 = stayed, 1 = churned  

**Size**:  
- **Train**: 165,034 rows × 14 columns (including the "Exited" target)  
- **Test**: 110,023 rows × 13 columns (without the target)  

**Split**:  
- 60% training  
- 20% validation  
- 20% testing


#### Preprocessing / Clean up

**Null values**: None found  
**Duplicates**: Identified after removing ID and CustomerID columns. Rows with identical surnames, balances, and estimated salaries were assumed to be duplicates and removed.  

**Categorical columns**:  
- `Geography` and `Gender` encoded via `OneHotEncoder` from scikit-learn.  

**Numerical columns**:  
- `CreditScore`, `Balance`, `Age`, `Tenure`, `EstimatedSalary`, and `NumOfProducts` scaled using `StandardScaler`.  
  - (Note: Both `MinMaxScaler` and `StandardScaler` were tested and yielded comparable results.)  
- Remaining numerical columns (e.g., `HasCrCard`, `IsActiveMember`) were already in binary format and left unchanged.


#### Data Visualization
A bar chart was created to visualize the distribution of the target variable. Here, an imbalance was observed with significantly more customers staying than churning.

<div align='center'>
  
  ![](Pictures/Imbalance.png)
  
</div>
  
After applying standard scaling, histograms of all features were plotted to compare their distributions across classes. Among the numerical features, **Age** emerged as a particularly strong separator between customers who stayed and those who churned.

<div align='center'>
  
  ![](Pictures/Features.png)

</div>


### Problem Formulation
   
The dataset contains customer features such as balance, creditscore, tenure and the likes. The goal is to predict churn rates for each customer, where 0 indicates the customer stayed and 1 indicates they churned. 
For the training dataset, the models were evaluated using the "Exited" column which gives the actual churn rates for each customers. In contrast, the test dataset does not include this column - the models must infer churn probabilities as part of the challenge submission.

  * Models
    * Decision Tree: chosen for its simplicity and history of giving good scores.
    * Histogram-based Gradient Boosting: selected for its robustness in providing fast and accurate results on large tabular datasets.  
    * Random Forest: used to leverage the power of multiple decision trees to attain potentially better scores. 
    * Logistic Regression: chosen because it's a simple yet effective baseline model for binary classification.
    * K-Nearest Neighbors: selected for a different approach to predicting based on local similarity rather than model learning.
  * Parameters:
    * All models were initiually configured with reasonable defaults (e.g. setting random state and class weight= 'balanced' where applicable).
    * Histogram-based Gradient Boosting was further fine-tuned using Scikit learn's Randomized Searched CV to find the optimal parameters for the model.
    * Full parameters settings and tuning details can be found in ML.ipynb notebook.

### Training

All of the machine learning algorithms mentioned above were implemented and trained in ML.ipynb notebook using the training dataset. Training times were minimal — under 1 minute for all models — due to the relatively small dataset size and the absence of deep learning computations.

### Performance Comparison

  The models were evaluated using standard classification metrics such as Accuracy, F1 Score, Precision, Recall, and most importantly, AUC Score (area under the ROC curve), which is the main evaluation metric for the Kaggle challenge.

  
<div align="center">

| Classifier           | Accuracy |    F1    | Precision |  Recall  | AUC Score |
|----------------------|----------|----------|-----------|----------|-----------|
| LogReg               | 0.752835 | 0.771438 | 0.4458    | 0.734278 | 0.818223  |
| RandomForest         | 0.817082 | 0.827643 | 0.544846  | 0.776348 | 0.883747  |
| HGB                  | 0.813565 | 0.824983 | 0.537952  | 0.786902 | 0.887020  |
| Decision Tree        | 0.813838 | 0.824218 | 0.539990  | 0.758421 | 0.871243  |
| K-Nearest Neighbors  | 0.844127 | 0.835958 | 0.668629  | 0.509036 | 0.819269  |

</div>

  
  ROC curves and its AUC scores were calculated for all models implemented into the training dataset at 60% split. Cross-validation tests were run to ensure that the scores were not biased or happened by chance.


<div align='center'>  
  
  ![](Pictures/ML_AUC.png)
  
</div>
 
  The Histogram-based Gradient Boosting model consistently outperformed others. It was selected for further tuning and was evaluated again using an 80-20 train-test split. Interestingly, increasing the training data slightly reduced its performance but the model still maintained the highest average AUC score among all tested models.


  <div align='center'>
    
  ![](Pictures/HGB_AUC.png)

  </div>


### Conclusions

Among the top-performing models, Histogram-based Gradient Boosting had a slight edge over Decision Tree and Random Forest in this dataset. However, all three performed in similar ranges for this problem. Therefore, simple models like Decision Tree can stil offer strong baseline performance without withholding too much potentials. 

### Future Work

Future improvements could include testing other boosting algorithms such as XGBoost, LightGBM, and CatBoost — all of which are known for their high performance in structured data challenges. These models have shown strong results in similar Kaggle competitions and could potentially push AUC scores even higher. For this particular challenge, one of the models that the top scorers used was LightGMBoost.


## How to reproduce results

 Reproducing results obtained in this repository can be done simply by following the workflow indicated below. This is preferably done in a Jupyter Notebook environment on a MacOS with python, scikit learn, and potentially the Kaggle API installed into Terminal. Other software setups and steps to train and evaluate model performances are further expanded in the notebooks included in this repository.

### Overview of files in repository

The files in this repository in Kaggle_Bank folder should be read from Data Load n Initial Look, Data Viz, Data Clean N Prep, ML, and Bank Churn Final to obtain a good understanding of the workflow. 
  * Data_Load_N_Initial_Look.ipynb: Downloads the bank churn dataset and explores its contents quickly.
  * Data_Viz.ipynb: Creates various visualizations of the data
  * Data_Clean_N_Prep.ipynb: Conducts preprocessing processes like deleting duplcations, onehot encoding variables as well as scaling numerical features.
  * ML.ipynb: Contains functions that build various machine learning models and evaluate their performances.
  * Bank_Churn_Final.ipynb: Contains all the preprocessing and machine learning processes necessary to obtain results and convert them into the proper submission format.
* Other files seen: 
  * submission.csv: Submission file created after all of the work is done. 
  * sample_submission.csv: A sample of the submission format provided by the challenge.
  * train.csv: Train dataset provided by the challenge.
  * test.csv: Test dataset provided by the challenge.

Pictures folder contains all images displayed throughout this readme file. 


### Software Setup
* Libraries like pandas, matplotlib, numpy, math, and scipy are needed and can be called using import. 
* Scikit learn needs to be downloaded into terminal using pip install before importing it into notebook.


### Data

 Data for this challenge can be downloaded through the [Kaggle Challenge website](https://www.kaggle.com/competitions/playground-series-s4e1/overview). Most convenitently, the API for Kaggle can be downloaded into Terminal which should then be relocated to the correct .kaggle folder. Then, use kaggle command to download datasets from the kaggle websites into jupyter notebook or any preferred environment. 

### Training

 Different models are initiated, fitted and trained on the train portion (60% of total data points) of train.csv dataset.

### Performance Evaluation

 Performance of the models can be evaluated via calculating the area under the ROC curve scores on the validation and test portions of the dataset. Cross-validation tests should also be run to ensure unbiased results. 


## Citations

 Kaggle Dataset: Walter Reade and Ashley Chow. Binary Classification with a Bank Churn Dataset . ([https://kaggle.com/competitions/playground-series-s4e1](https://kaggle.com/competitions/playground-series-s4e1)), 2024. Kaggle.







