# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


def Random_Forest_no_hp(df_train, df_test):
    #Splitting Datasets into X and y
    X_train_rf = df_train.drop('Type', axis =1)
    y_train_rf = df_train['Type']
    
    X_test_rf = df_test.drop('Type', axis =1)
    y_test_rf = df_test['Type']
    
    #Initializing model
    rf_model= RandomForestClassifier()
        
    #Training model
    rf_model.fit(X_train_rf,y_train_rf)
    
    #Predicting values
    y_pred_rf = rf_model.predict(X_test_rf)
    
    # Evaluation
    accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
    precision_rf = precision_score(y_test_rf, y_pred_rf)
    recall_rf = recall_score(y_test_rf, y_pred_rf)
    
    return rf_model, X_test_rf, y_test_rf, y_pred_rf, accuracy_rf, precision_rf, recall_rf