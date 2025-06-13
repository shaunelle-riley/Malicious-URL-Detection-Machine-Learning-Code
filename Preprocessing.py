#-*- coding: utf-8 -*-
"""
Created on Fri Jun  6 14:47:07 2025

@author: sqril
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

#simple function is appropriate given the small number of values and no incosisten variations
def charset_cleaner(charset):
    if pd.isna(charset): 
        return "null"
    
    charset = str(charset).strip().lower()
    if charset not in ['utf-8','iso-8859-1','us-ascii']:
        return "other"
    else:
        return charset
    
def keep_null_country(country):
    if pd.isna(country):
        return "NULL"
    
    else: 
        return country    
    
    
def get_common_countries(URL_data, country_col = "WHOIS_COUNTRY", threshold=3):
    URL_data[country_col] = URL_data[country_col].apply(keep_null_country)        
    
   
    URL_data[country_col] = URL_data[country_col].str.upper()
    
    def country(country):
        if country in ["[U'GB'; U'UK']", "UNITED KINGDOM", "GREAT BRITAIN", 'UK']:
            return "GB"
        
        else:
            return country
        
    URL_data[country_col] = URL_data[country_col].apply(country)  
    
    common_count = URL_data[country_col].value_counts()
    common_countries = common_count[common_count > threshold].index.tolist()
    
    if "NULL" not in common_countries:
        common_countries.append("NULL")
        
    if "RU" not in common_countries:
        common_countries.append("RU")

    return common_countries


    
def country_cleaner(URL_data, country_col='WHOIS_COUNTRY', common_countries=None):
    URL_data[country_col] = URL_data[country_col].apply(
        lambda x: x if x in common_countries else 'OTHER'
    )
    return URL_data[country_col]


def keep_null_state(state):
    if pd.isna(state):
        return "NULL"
    
    else:
        return state


def get_common_states(URL_data, state_col = 'WHOIS_STATEPRO', threshold = 3):
    URL_data[state_col] = URL_data[state_col].apply(keep_null_state) 
    
    URL_data[state_col] = URL_data[state_col].str.upper()
    
    #creating dictionary for states
    state_dict = {'CA':'CALIFORNIA', 'AZ':'ARIZONA', 'OH':'OHIO', 'UT':'UTAH', 'WA':'WASHINGTON', 'NY':'NEW YORK', 'TX':'TEXAS', 
                  'MO':'MISSOURI', 'VA':'VIRGINIA', 'DC':'WASHINGTON', 'IL':'ILLINOIS', 'FL':'FLORIDA', 'PA':'PENNSYLVANIA', 
                  'OR':'OREGON', 'KS':'KANSAS', 'WC1N':'WISCONSIN', 'WI':'WISCONSIN','NJ':'NEW JERSEY', 'ON':'ONTARIO', 'DE':'DENVER',  
                  'NC':'NORTH CAROLINA', 'MA': 'MASSACHUSETTS', 'QLD':'QUEENSLAND', 'BC': 'BRITISH COLUMBIA', 'CO':'COLORADO',
                  'GA':'GEORGIA', 'LA':'LOS ANGELES', 'MI':'MICHIGAN', 'NV':'NEVADA', 'UK':'UNITED KINGDOM', 'NSW':'NEW SOUTH WALES'}
    
    URL_data['WHOIS_STATEPRO'] = URL_data['WHOIS_STATEPRO'].replace(state_dict)
    
    #creating common states list
    common_count = URL_data[state_col].value_counts()
    common_states = common_count[common_count > threshold].index.tolist()
    
    if "NULL" not in common_states:
        common_states.append("NULL")
        
    return common_states
    

def state_cleaner(URL_data, state_col = 'WHOIS_STATEPRO', common_states=None):    
    #applying state cleaner and updating dataframe
    URL_data[state_col] = URL_data[state_col].apply(
        lambda x: x if x in common_states else 'Other'
    )
    return URL_data[state_col]


def keep_null_server(server):
    if pd.isna(server):
        return "NULL"
    
    else:
        return server



def get_common_servers(URL_data, server_col = "SERVER", threshold=3):
    URL_data[server_col] = URL_data[server_col].apply(keep_null_server)
    
    URL_data[server_col] = URL_data[server_col].str.upper()
    
    common_count = URL_data[server_col].value_counts()
    common_servers = common_count[common_count > threshold].index.tolist()
    
    if "NULL" not in common_servers:
        common_servers.append("NULL")
        
    return common_servers

    
def server_cleaner(URL_data, server_col='SERVER', common_servers=None):
    URL_data[server_col] = URL_data[server_col].apply(
        lambda x: x if x in common_servers else 'Other'
    )
    return URL_data[server_col]


def create_numerical_df(URL_data):
    num_URL_data = URL_data.select_dtypes(include=['float64', 'int64'])
    
    label_cols = ['CHARSET', 'SERVER', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO']
    label_cols_name = ['CHARSET_num', 'SERVER_num', 'WHOIS_COUNTRY_num', 'WHOIS_STATEPRO_num']
    le = LabelEncoder()
    
    for col, col_name in zip(label_cols, label_cols_name):
        if col in URL_data.columns:
            num_URL_data[col_name] = le.fit_transform(URL_data[col].astype(str))
    
        
    return num_URL_data    
    


def Preprocess(URL_data, common_countries=None, common_states=None, common_servers=None, 
               countries_threshold=3, states_threhold = 3, servers_threshold=3, is_train=True):
    
    URL_data = URL_data.drop(["URL","CONTENT_LENGTH", 'WHOIS_REGDATE','WHOIS_UPDATED_DATE'], axis=1)
    
    URL_data.isnull().sum()
    
    #applying charset cleaner    
    URL_data['CHARSET'] = URL_data['CHARSET'].apply(charset_cleaner)
    URL_data['CHARSET'].value_counts() #verifying charset cleaner was applied correctly
    
    #applying countrry cleaner
    if is_train:
        common_countries = get_common_countries(URL_data, threshold=countries_threshold)
    URL_data['WHOIS_COUNTRY'] = country_cleaner(URL_data, common_countries=common_countries)
    URL_data['WHOIS_COUNTRY'].value_counts() #verifying country cleaner was applied correctly
    
    #applying state cleaner
    if is_train:
        common_states = get_common_states(URL_data, threshold=states_threhold)
    URL_data['WHOIS_STATEPRO'] = state_cleaner(URL_data, common_states=common_states)
    URL_data['WHOIS_STATEPRO'].value_counts()
    
    #applying server cleaner
    if is_train:
        common_servers = get_common_servers(URL_data, threshold=servers_threshold)
    URL_data['SERVER'] = server_cleaner(URL_data, common_servers=common_servers)
    URL_data['SERVER'].value_counts()
    
    #dropping null values from URL_data
    URL_data.isnull().sum()
    URL_data.dropna(inplace=True)
    
    #creating numerical dataframe
    num_URL_data = create_numerical_df(URL_data)
    
    #match indexes for both df
    num_URL_data.index = URL_data.index
        
    return URL_data, num_URL_data, common_countries, common_states, common_servers


if __name__ == "__main__":
    '''Importing data'''
    URL_data = pd.read_csv("URL_dataset.csv")
    processed_df, num_df, common_countries, common_states, common_servers = Preprocess(URL_data)
