# -*- coding: utf-8 -*-


import pandas as pd
from Dataset_Info import Info 
from Outlier_Analysis import Outlier_Analysis
from Preprocessing import Preprocess
from Splitting_Dataset import SplittingDataset
from Exploratory_Analysis import Exploratory_Analysis as Explore
from FitScalers import FitScalers
from TransformScalers import TransformScalers
from RandomForest import Random_Forest
from XGBoost import XGBoost
from XGBoost_No_hp import XGBoost_no_hp
from RandomForest_No_hp import Random_Forest_no_hp
from Evaluate_model import EvaluateModel as Evaluate
from Metrics_Table import metrics_table


#importing dataset
df = pd.read_csv("URL_dataset.csv")


#getting information about dataset
Info(df)


#Splitting the dataset into training and testing - scale and normailize dataset
df_train, df_test = SplittingDataset(df)


'''Training Dataset'''
#Preprocessing data - standardizing categorical values and creating numerical dataframe
df_processed, df_num, common_countries, common_states, common_servers = Preprocess(df_train)
df_processed.head()
df_num.head()


#Doing Exploratory Analysis on Training dataset before outlier removal
Explore(df_processed,df_num)


#Outlier analysis DID NOT REMOVE OUTLIERS DUE TO THE HEAVILY SKEWED DISTRIBUTION AND LARGE NUMBER OF OUTLIERS
Outlier_Analysis(df_processed, df_num)


#Scaling dataset
df_num_scaled, min_max_scaler = FitScalers(df_num)



'''Testing Dataset'''
df_test_processed, df_test_num, _, _,_ = Preprocess(
    df_test, 
    common_countries=common_countries, 
    common_states=common_states, 
    common_servers=common_servers, 
    is_train=False)
    
df_test_scaled = TransformScalers(df_test_num, min_max_scaler)



'''Evaluation'''
#XGBoost with hyperparameters
xgb_model, X_test_xgb, y_test_xgb, y_pred_xgb, accuracy_xgb, precision_xgb, recall_xgb = XGBoost(df_num_scaled, df_test_scaled)

Evaluate(xgb_model, X_test_xgb, y_test_xgb, y_pred_xgb)


#Random Forest
rf_model, X_test_rf, y_test_rf, y_pred_rf, accuracy_rf, precision_rf, recall_rf = Random_Forest(df_num_scaled, df_test_scaled)

Evaluate(rf_model, X_test_rf, y_test_rf, y_pred_rf)


#XGBoost without hyperparameters
xgb_model_no_hp, X_test_xgb_no_hp, y_test_xgb_no_hp, y_pred_xgb_no_hp, accuracy_xgb_no_hp, precision_xgb_no_hp, recall_xgb_no_hp = XGBoost_no_hp(df_num_scaled, df_test_scaled)

Evaluate(xgb_model_no_hp, X_test_xgb_no_hp, y_test_xgb_no_hp, y_pred_xgb_no_hp, no_hp= True)


#Random Forest without hyperparameters
rf_model_no_hp, X_test_rf_no_hp, y_test_rf_no_hp, y_pred_rf_no_hp, accuracy_rf_no_hp, precision_rf_no_hp, recall_rf_no_hp = Random_Forest_no_hp(df_num_scaled, df_test_scaled)

Evaluate(rf_model_no_hp, X_test_rf_no_hp, y_test_rf_no_hp, y_pred_rf_no_hp, no_hp= True)


#Displaying all Metrics
data = {
    "Classification Algorithms": [
        "XGBoost", 
        "XGBoost (No Hyperparameters)", 
        "Random Forest", 
        "Random Forest (No Hyperparameters)"
    ],
    "Accuracy Score": [
        accuracy_xgb, 
        accuracy_xgb_no_hp, 
        accuracy_rf, 
        accuracy_rf_no_hp
    ],
    "Precision Score": [
        precision_xgb, 
        precision_xgb_no_hp, 
        precision_rf, 
        precision_rf_no_hp
    ],
    "Recalll Score": [
        recall_xgb, 
        recall_xgb_no_hp, 
        recall_rf, 
        recall_rf_no_hp
    ]
}

all_metrics = pd.DataFrame(data)
all_metrics.set_index('Classification Algorithms', inplace=True)
metrics_table(all_metrics)

#Final Metrics
final_data = {
    "Classification Algorithms": [
        "XGBoost", 
        "Random Forest"
    ],
    "Accuracy Score": [
        accuracy_xgb,
        accuracy_rf
    ],
    "Precision Score": [
        precision_xgb,  
        precision_rf
    ],
    "Recalll Score": [
        recall_xgb,
        recall_rf
    ]
}

final_metrics = pd.DataFrame(final_data)
final_metrics.set_index('Classification Algorithms', inplace=True)
metrics_table(final_metrics)

'''Complete Dataset for training and testing'''
df_num_scaled['Type'].value_counts()
df_test_scaled['Type'].value_counts()
combined_df = pd.concat([df_num_scaled,df_test_scaled], ignore_index=True)
combined_df.shape
combined_df['Type'].value_counts()
