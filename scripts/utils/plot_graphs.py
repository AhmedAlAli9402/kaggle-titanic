import pandas as pd
import matplotlib.pyplot as plt

def plot_survived_vs_pclass(train_df):
    plt.figure(figsize=(10, 5))
    survived_counts = train_df[train_df['Survived'] == 1]['Pclass'].value_counts()
    survived_counts.plot(kind='bar')
    plt.title('Pclass vs Survived')
    plt.xlabel('Pclass')
    plt.ylabel('Count')
    plt.show()