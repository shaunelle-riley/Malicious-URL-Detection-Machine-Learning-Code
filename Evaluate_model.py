# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def ClassificationReportTable(model_name, y_test, y_pred):
    
    plt.rcParams['font.family'] = 'Calibri'  
    plt.rcParams['font.size'] = 12
    
    #Generating the classification report as a dictionary and converting to DataFrame
    report_dict = classification_report(y_test, y_pred, target_names=['Benign', 'Malicious'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(3)
    
    #Creating the figure and axes
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off') 
    
    #Creating the table
    table = ax.table(
        cellText=report_df.values,
        rowLabels=report_df.index,
        colLabels=report_df.columns,
        loc='center',
        cellLoc='center'
    )
    
    #Formating the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    #Creating title
    plt.title(f"{model_name} Classification Report", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    


def ConfusionMatrix(model_name, y_test, y_pred):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'])
    plt.title(f'{model_name} Confusion Matrix', fontsize=12)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    
def ConfusionMatrix_FeatImpo(model_name, model, X_test, y_test, y_pred):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'],
                ax=ax1)
    ax1.set_title(f'{model_name} Confusion Matrix', fontsize = 12)
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Feature Importance (top 5 only)
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_.round(3)
        }).sort_values('importance', ascending=False).head(8)
    
    sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax2)
    ax2.tick_params(axis='y', labelsize=10)
    ax2.set_title('Top 5 Important Features')
    for c in ax2.containers:
        ax2.bar_label(c, label_type='center', fontsize =10)
    
    plt.tight_layout()
    plt.show()
    
def EvaluateModel(model, X_test, y_test, y_pred, no_hp = False):
    # Get the model's class name
    model_name = model.__class__.__name__
    
    #Evaluating Model
    roc_auc = roc_auc_score(y_test, y_pred)
    
    if no_hp:
        model_name += ' (No Hyperparameters)'
    
    '''
    print(f"{model_name} roc-auc: {roc_auc:.2f}")
    
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))
    '''
    
    if model_name in ('RandomForestClassifier', 'DecisionTreeClassifier', 'XGBClassifier', 
                      'RandomForestClassifier (No Hyperparameters)',
                      'DecisionTreeClassifier (No Hyperparameters)', 'XGBClassifier (No Hyperparameters)'):
        
        #Displaying Classification Report Table
        ClassificationReportTable(model_name, y_test, y_pred)
        
        #Displaying Confusion Matrix and Feature Importance
        ConfusionMatrix_FeatImpo(model_name, model, X_test, y_test, y_pred)
        
    else:
        #Displaying Classification Report Table
        ClassificationReportTable(model_name, y_test, y_pred)
        
        #Displaying Confusion Matrix
        ConfusionMatrix(model_name, y_test, y_pred)
        
        
        
        
        '''# Create visualizations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign', 'Malicious'],
                    yticklabels=['Benign', 'Malicious'],
                    ax=ax1)
        ax1.set_title(f'{model_name} Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Feature Importance (top 5 only)
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_.round(3)
            }).sort_values('importance', ascending=False).head(8)
        
        sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax2)
        ax2.tick_params(axis='y', labelsize=10)
        ax2.set_title('Top 5 Important Features')
        for c in ax2.containers:
            ax2.bar_label(c, label_type='center', fontsize =16)
        
        plt.tight_layout()
        plt.show()
        
    else:
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Benign', 'Malicious'],
                    yticklabels=['Benign', 'Malicious'])
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')'''
            