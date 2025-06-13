import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#columns not included in outlier analysis
exclude_cols = ['Type', 'CHARSET_num', 'SERVER_num', 'WHOIS_COUNTRY_num', 'WHOIS_STATEPRO_num']


def plot_box_plot(num_URL_data):
        
    #excluding categorical values    
    columns_to_plot = [col for col in num_URL_data.columns if col not in exclude_cols]
    
    #Creating boxplots for outlier analysis
    for col in columns_to_plot:
        sns.boxplot(x=num_URL_data[col] , data=num_URL_data)
        plt.title(f"Boxplot of {col}")
        plt.show()
        
        sns.displot(data=num_URL_data, x=col, kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()



def outlier_percentile(num_URL_data, lower_percentile , upper_percentile):
    outliers_dict = {}
    outliers_combined = []
    
    
    if lower_percentile==0.25 and upper_percentile==0.75:
        q1 = num_URL_data.quantile(.25)
        q3 = num_URL_data.quantile(.75)
        iqr = q3 - q1
        
        lowerBound=q1-1.5*iqr
        upperBound=q3+1.5*iqr

    else:
        lowerBound = num_URL_data.quantile(lower_percentile)
        upperBound = num_URL_data.quantile(upper_percentile)
    
    
    #excluding categorical values    
    columns_to_calculate = [col for col in num_URL_data.columns if col not in exclude_cols]
    
    #calculating outlier per column
    for col in columns_to_calculate:
        outliers = np.where((num_URL_data[col] < lowerBound[col]) | (num_URL_data[col] > upperBound[col]))[0]
        outliers_dict[col] = len(outliers)
        outliers_combined.extend(outliers)
        
    outliers_combined = np.unique(outliers_combined)
    
    return  lowerBound, upperBound, outliers_dict, outliers_combined


def plot_outliers(outliers_dict):
    
    # Convert dictionary to a Pandas Series
    outlier_series = pd.Series(outliers_dict).sort_values(ascending=False)

    #plotting outlier count
    plt.figure(figsize=(10, 6))
    sns.barplot(x=outlier_series.values, y=outlier_series.index, palette="viridis")
    plt.xlabel("Number of Outliers")
    plt.ylabel("Feature")
    plt.title("Outlier Counts per Feature")
    plt.tight_layout()
    plt.show()
        


def remove_outliers(num_URL_data, outlier_combined):
        
    #dropping outliers
    index_labels_to_drop = num_URL_data.iloc[outlier_combined].index
    num_URL_data_clean = num_URL_data.drop(index=index_labels_to_drop)
        
    return num_URL_data_clean
    
 
    
def Outlier_Analysis(URL_data, num_URL_data, plot = True, lower_percentile=0.01, upper_percentile = 0.99):
    original_shape = num_URL_data.shape
    
    if plot:
        #plotting boxplots and density plots to view outliers and distribution
        plot_box_plot(num_URL_data)
    
    #deciding percentile to remove outliers
    lowerBound, upperBound, outliers_dict, outliers_combined = outlier_percentile(
        num_URL_data, 
        lower_percentile= lower_percentile,
        upper_percentile= upper_percentile
        )
    
    #plotting outlier count
    plot_outliers(outliers_dict)
    
    #removing outliers
    num_URL_data_clean = remove_outliers(num_URL_data, outliers_combined)
    URL_data_clean = remove_outliers(URL_data, outliers_combined)
    cleaned_shape = num_URL_data_clean.shape
    
    #Outlier removal summary
    rows_dropped = original_shape[0] - cleaned_shape[0]
    print("\nOutlier removal complete.")
    print(f"\nOriginal number of rows: {original_shape[0]}")
    print(f"Remaining rows after outlier removal {num_URL_data_clean.shape[0]}")
    print(f"Total rows dropped: {rows_dropped} ({rows_dropped/original_shape[0]*100:.2f}% of data)")
    
    return URL_data_clean, num_URL_data_clean