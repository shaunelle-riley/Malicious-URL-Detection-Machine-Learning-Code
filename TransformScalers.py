# -*- coding: utf-8 -*-

import pandas as pd

def TransformScalers(df_test, min_max_scaler):
    
    data_scaled = min_max_scaler.transform(df_test)
    
    return pd.DataFrame(data_scaled, columns=df_test.columns, index=df_test.index)