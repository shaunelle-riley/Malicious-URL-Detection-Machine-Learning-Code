import pandas as pd
from sklearn.model_selection import train_test_split

#splitting Dataset to ensure all models use the same dataset

def SplittingDataset(df):
    
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['Type'],axis=1), df['Type'], stratify=df['Type'], test_size=0.2, random_state=101)
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    
    return df_train, df_test