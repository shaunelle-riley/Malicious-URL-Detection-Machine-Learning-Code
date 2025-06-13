# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import MinMaxScaler 

def FitScalers(num_URL_data):
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
           
    #apply min max scaler
    num_URL_data_scaled = min_max_scaler.fit_transform(num_URL_data)
    
    return pd.DataFrame(num_URL_data_scaled, columns=num_URL_data.columns, index=num_URL_data.index), min_max_scaler