# -*- coding: utf-8 -*-

from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve
from collections import Counter
from sklearn.model_selection import GridSearchCV
import numpy as np

def XGBoost_no_hp(df_train, df_test):
    #Splitting Datasets into X and y
    X_train_xgb = df_train.drop('Type', axis =1)
    y_train_xgb = df_train['Type']
    
    X_test_xgb = df_test.drop('Type', axis =1)
    y_test_xgb = df_test['Type']
    
    
    #Initializing Model
    xgb_model = XGBClassifier()
        
    #training model
    xgb_model.fit(X_train_xgb, y_train_xgb)
    
    
    #Predicting probabilities
    y_pred_xgb = xgb_model.predict(X_test_xgb)
    
        
    # Evaluation
    accuracy_xgb = accuracy_score(y_test_xgb, y_pred_xgb)
    precision_xgb = precision_score(y_test_xgb, y_pred_xgb)
    recall_xgb = recall_score(y_test_xgb, y_pred_xgb)
    
    return xgb_model, X_test_xgb, y_test_xgb, y_pred_xgb, accuracy_xgb, precision_xgb, recall_xgb

