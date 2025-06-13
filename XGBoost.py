# -*- coding: utf-8 -*-

from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve
from collections import Counter
from sklearn.model_selection import GridSearchCV
import numpy as np

def XGBoost(df_train, df_test):
    #Splitting Datasets into X and y
    X_train_xgb = df_train.drop('Type', axis =1)
    y_train_xgb = df_train['Type']
    
    X_test_xgb = df_test.drop('Type', axis =1)
    y_test_xgb = df_test['Type']
    
    #Counting Class Distribution
    counter = Counter(y_train_xgb)
    scale_pos_weight = counter[0] / counter[1]
    
    #Creating parameters
    params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1],
        'scale_pos_weight': [scale_pos_weight]  
    }
    
    #Grid Search with F1-score
    xgb_model = XGBClassifier()
    xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=params, scoring='f1', cv=3, verbose=1)
    
    #training model
    xgb_grid.fit(X_train_xgb, y_train_xgb)
    
    
    #Predicting probabilities
    y_probs = xgb_grid.predict_proba(X_test_xgb)[:, 1]
    
    #Threshold tuning: find best F1-score
    precision, recall, thresholds = precision_recall_curve(y_test_xgb, y_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-6)  # Avoid division by zero
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    #Predict using best threshold
    y_pred_adjusted_xgb = (y_probs >= best_threshold).astype(int)
    
    # Evaluation
    accuracy_xgb = accuracy_score(y_test_xgb, y_pred_adjusted_xgb)
    precision_xgb = precision_score(y_test_xgb, y_pred_adjusted_xgb)
    recall_xgb = recall_score(y_test_xgb, y_pred_adjusted_xgb)
    
    return xgb_grid.best_estimator_, X_test_xgb, y_test_xgb, y_pred_adjusted_xgb, accuracy_xgb, precision_xgb, recall_xgb

