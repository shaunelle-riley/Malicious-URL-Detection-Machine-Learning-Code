# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns


'''Plotting Attribute Averages by Type'''
def plot_att_avgs_by_type(data, value_cols, group_col='Type Name' ):
    for val_col in value_cols:
        #creating dataset for plot
        avg_data = data.groupby(group_col)[val_col].mean().round(3).reset_index()
        
        #plotting data
        plt.figure(figsize=(13,9))
        ax = sns.barplot(data = avg_data, x = group_col, y = val_col, palette = 'crest')

        #Labels and Titles
        plt.xticks(fontsize=12)
        ax.set_xlabel('Type', fontsize = 14)
        ax.set_ylabel(f'Average {val_col} ', fontsize = 14)
        ax.set_title(f'Average {val_col} by Type', fontsize = 22)

        for c in ax.containers:
            ax.bar_label(c, label_type='center', fontsize =16)
        
        plt.tight_layout()        
        plt.show()
        
        
def plot_type_count_by_att(data, group_cols, value_col = 'Type Name', max_label_len=22):
    for group_col in group_cols:
        group_data = data.groupby(group_col)[value_col].value_counts().reset_index()
        
        #Truncating long labels
        group_data[group_col] = group_data[group_col].astype(str).str.slice(0, max_label_len)
        
                #plotting data
        plt.figure(figsize=(13,9))
        ax = sns.barplot(data = group_data, x=group_col, y = 'count', hue = value_col, palette = 'viridis')
        
        if group_col in ['SERVER']:
            fontsize=9
        else:
            fontsize =12
               
        #labels
        plt.xticks(fontsize=fontsize, rotation=75)
        plt.yticks(fontsize=fontsize)
        ax.set_xlabel(f'{group_col}', fontsize = 14)
        ax.set_ylabel('Count of Type', fontsize = 14)
        ax.set_title(f'Count of Type by {group_col}', fontsize=22)
        plt.legend(ncol=2, loc='center',bbox_to_anchor=(0.5, 1.1))
        
        for c in ax.containers:
            ax.bar_label(c, fontsize =10)

        plt.tight_layout()
        plt.show()
    

def Exploratory_Analysis(URL_data, num_URL_data):
    URL_data["Type Name"] = URL_data["Type"].map({0: "Benign", 1: "Malicious"})
    URL_type_count = URL_data['Type Name'].value_counts()


    #URL type distribution pie chart
    fig, ax = plt.subplots()
    ax.pie(URL_type_count, labels=URL_type_count.index, autopct='%0.1f%%')
    ax.set_title('URL Type Distribution')
    plt.show()


    #plotting numerical attributes averages by type
    plot_att_avgs_by_type(
        data = URL_data, 
        value_cols = ['URL_LENGTH', 'NUMBER_SPECIAL_CHARACTERS', 'TCP_CONVERSATION_EXCHANGE',
               'DIST_REMOTE_TCP_PORT', 'REMOTE_IPS', 'APP_BYTES', 'SOURCE_APP_PACKETS',
               'REMOTE_APP_PACKETS', 'SOURCE_APP_BYTES', 'REMOTE_APP_BYTES',
               'APP_PACKETS', 'DNS_QUERY_TIMES'])
    
    
    #plotting count of each type for categorical attributes
    plot_type_count_by_att(
        data = URL_data, 
        group_cols = ['CHARSET', 'SERVER', 'WHOIS_COUNTRY', 'WHOIS_STATEPRO'])
    

    #heatmap
    plt.figure(figsize=(12,10))
    sns.heatmap(num_URL_data.corr(),annot=True,cmap="Greens")
    plt.show()




