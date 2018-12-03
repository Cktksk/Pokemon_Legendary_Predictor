import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Plot stat vs legendary
    data = pd.read_csv("Pokemon.csv")
    stat = data.drop(['#','Type 1', 'Type 2','Total','Generation','Name'], axis = 1)
    stat = pd.melt(stat, id_vars = ['Legendary'], var_name = "stat")

    plt.figure()
    sns.swarmplot(x="stat", y="value", data=stat, hue="Legendary").get_figure().savefig("Results//Stat_vs_Le.png")

    # Plot type vs legendary
    stat1 = data[['Type 1', 'Legendary']]
    stat2 = data[['Type 2', 'Legendary']]
    typ = data['Type 2'].unique()
    dic_type = {typ[i] : i for i in range(19)}
    val = [[typ[i], 0, 0, 0] for i in range(19)]

    for i in range(800):
        val[dic_type[stat1.values[i][0]]][3] += 1
        if stat1.values[i][1] == True:
            val[dic_type[stat1.values[i][0]]][1] += 1
            val[dic_type[stat2.values[i][0]]][2] += 1
        

    df = pd.DataFrame(val, columns = ['Type Name', 'Type 1', 'Type 2', 'Total'])
    plt.figure()
    sns.barplot(x="Type Name", y="Type 1", data=df).get_figure().savefig("Results//Type1_vs_Le.png")
    plt.figure()
    sns.barplot(x="Type Name", y="Type 2", data=df).get_figure().savefig("Results//Type2_vs_Le.png")