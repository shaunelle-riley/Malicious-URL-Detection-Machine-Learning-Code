# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def metrics_table(metrics):
    
    plt.rcParams['font.family'] = 'Calibri'  
    plt.rcParams['font.size'] = 12
    
    #Plot as a table
    fig, ax = plt.subplots(figsize=(8, 2.5))  
    ax.axis('off')  

    # Create table
    table = ax.table(
        cellText=metrics.round(2).values,
        rowLabels=metrics.index,
        colLabels=metrics.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)  

    plt.title("Model Performance Comparison",  fontsize=12)
    plt.tight_layout()
    plt.show()
