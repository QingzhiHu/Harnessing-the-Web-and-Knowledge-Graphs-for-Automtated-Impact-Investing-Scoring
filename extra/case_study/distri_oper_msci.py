import pandas as pd
msci = pd.read_csv("./data/msci.csv")
msci2 = pd.read_csv("./data/msci2.csv").rename(columns={"SDG_03_OPS_ALIGNMENT":"SDG_03_OPER_ALIGNMENT"})
companies = pd.read_csv("results_news/plot.csv").company.unique()

# companies = msci[["Company Name", "SDG_01_NET_ALIGNMENT_SCORE"]].dropna()["Company Name"].values
tmp = msci2.merge(msci, left_on="Figi", right_on="Company ID")
# tmp = SDG1.merge(SDG2, left_on="Company ID", right_on="Figi")

tmp = tmp[tmp["Company Name"].isin(companies)]

# ["SDG_01_PROD_ALIGNMENT", "SDG_02_PROD_ALIGNMENT", ]
import numpy as np
from matplotlib import pyplot as plt
def count_ranges(column_name):
    very_positive = 0
    positive = 0
    neutral = 0
    negative = 0
    very_negative = 0

    for element in tmp[column_name].dropna().values:
        if element == "Strongly Aligned":
            very_positive += 1

        if element == "Aligned":
            positive += 1

        if element == "Neutral":
            neutral += 1

        if element == "Misaligned":
            negative += 1

        if element == "Strongly Misaligned":
            very_negative += 1

    return [very_positive, positive, neutral, negative, very_negative]

columns = ['SDG_01_OPER_ALIGNMENT','SDG_02_OPER_ALIGNMENT', 'SDG_03_OPER_ALIGNMENT',
 'SDG_04_OPER_ALIGNMENT', 'SDG_05_OPER_ALIGNMENT','SDG_06_OPER_ALIGNMENT',
 'SDG_07_OPER_ALIGNMENT','SDG_08_OPER_ALIGNMENT', 'SDG_09_OPER_ALIGNMENT',
 'SDG_10_OPER_ALIGNMENT', 'SDG_11_OPER_ALIGNMENT','SDG_12_OPER_ALIGNMENT',
 'SDG_13_OPER_ALIGNMENT','SDG_14_OPER_ALIGNMENT', 'SDG_15_OPER_ALIGNMENT',
 'SDG_16_OPER_ALIGNMENT', 'SDG_17_OPER_ALIGNMENT']

range_values_all = []
for x in columns:
    range_values = count_ranges(x)
    range_values_all.append(range_values)

df_filter = pd.DataFrame(range_values_all).T
df_filter.columns = columns
# variables
labels = ['very positive [+5:+10]', 'positive [+1:+5)', 'neutral (-1:+1)', 'negative (-5:-1])', 'very negative [-5, -10]']
colors = ['#1D2F6F', '#8390FA', 'gray', '#FAC748', 'cyan']
title = 'SDG distributions\n'
subtitle = 'Stacked bar-chart'
def plot_stackedbar_p(df, labels, colors, title, subtitle):
    fields = df.columns.tolist()

    # figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 12))
    # plot bars
    left = len(df) * [0]
    for idx, name in enumerate(fields):
        # print(idx, name)
        plt.barh(df.index,
                 df[name],
                 left = left,
                 color=colors[idx])
        left = left + df[name]
    # title and subtitle
    plt.title(title, loc='left')
    plt.text(0, ax.get_yticks()[-1] + 0.75, subtitle)
    # legend
    plt.legend(labels, bbox_to_anchor=([0.58, 1, 0, 0]), ncol=2, frameon=False)
    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # format x ticks
    xticks = np.arange(0,1.1,0.1)
    xlabels = ['{}%'.format(i) for i in np.arange(0,101,10)]
    plt.xticks(xticks, xlabels)
    # adjust limits and draw grid lines
    plt.ylim(-0.5, ax.get_yticks()[-1] + 0.5)
    ax.xaxis.grid(color='gray', linestyle='dashed')
    plt.show()

plot_stackedbar_p(df_filter.T, labels, colors, title, subtitle)   
