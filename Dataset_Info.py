# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns

def Info(URL_data):
    
    print("\nDataset shape: ",URL_data.shape)
    print("\nDataset Columns:\n",URL_data.columns)
    print("\nDataset Sum of Null Values:\n"+ URL_data.isnull().sum().to_string())
    print("\nDataset Info:")
    URL_data.info()
    print("\nDataset head:\n",URL_data.head())
    print("\nDataset tail:\n",URL_data.tail())
  