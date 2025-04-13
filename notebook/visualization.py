import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Visualization:

    def __init__(self, df):
        self.df = df

    def barplot_numeric(self, columns):

        for col in columns:
            plt.figure(figsize=(10,6))
            sns.histplot(
                data=self.df,
                bins=30,
                kde=True,
                x=col
            )
            plt.title(f'Bar Plot for {col}', fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
            plt.close()


