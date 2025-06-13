# Malicious-URL-Detection-Machine-Learning-Code
Creating an XGBoost model and a Random Forest model to detect malicious URLs

This project shows the steps necessary to create robust XGBoost and Random Forest models to detect malicious URLs. The code is separated to give more of a automated pipeline effect. 

The folder contains 15 files:
1. URL_dataset - the dataset which the models were trained and tested on
2. Dataset_Info - This short file is used to give an initial overview of the dataset
3. Splitting_Dataset - Used to split the dataset to ensure there is no leakage of test data information into the Preprocess process or the Exploratory Analysis.
4. Preprocessing - dimensionality reduction, handles null values, formatting for uniformed data, and Reduction of attribute values for categorical data, create numerical dataset
5. Exploratory_Analysis - used to find trends and patterns in the data
6. Outlier_Analysis - while created was not utilized as it negatively affected results and there was no concrete evidence show that extremely high values were incorrect.
7. FitScalers - used to scale the training data between 0 and 1
8. TransformScalers - used to transform test data between 0 and 1 using the scaler created by training data
9. XGBoostNo_hp - xgboost model with no hyperparameters
10. RandomForest_No - random forest model with no hyperparameters
11. XGBoost - xgboost model with hyperparameter tuning and threhold tuning
12. RandomForest - random forest model with hyperparameters
13. Evaluate_model - displays classification report, confusion matrix and feature importance
14. Metrics_Table - displays metrics as a table
15. Main - shows an example of the workflow  
