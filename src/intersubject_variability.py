import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sex = ['F','F','F','M','F','M','F','M','F','F','M','M','F','M',
          'M','M','M','M','F','F','M','F','M','M','M','M','F','F',
          'M','M','M','M','F','M','F','M','F','M','M','F','M','M']

left_handed = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,
               0,0,1,0,0,0,0,1,1,0,0,0,0,0,
               0,0,0,0,0,0,0,0,0,0,0,1,0,0]

order = ['Cued first','Uncued first']*21

subjects=['pilot3','136','358','916','347','766','661','959','205','400',
          '756','897','420','804','207','196','295','402','722','720',
          '966','568','482','307','730','228','785','640','457','183',
          '759','278','732','441','930','755','689','877','312','751',
          '443','223']

ages = [26,25,29,23,33,25,25,26,27,26,25,24,
        20,25,24,30,28,25,40,25,26,42,28,29,23,28,25,
        27,39,23,29,64,25,22,58,60,53,45,22,25,31,23]

data = pd.read_csv('./test_set_performance_4-8Hz_0.csv')

data['handedness']  = np.nan
data['sex'] = np.nan
data['order'] = np.nan
data['age'] = np.nan

for i,s in enumerate(subjects):
    data.loc[data.subject==s, 'handedness'] = left_handed[i]*'Left' + (1-left_handed[i])*'Right'
    data.loc[data.subject==s, 'order'] = order[i]
    data.loc[data.subject==s, 'sex'] = sex[i]
    data.loc[data.subject==s, 'age'] = ages[i]
    
#%% then on to the tests
from scipy.stats import ttest_ind, pearsonr
sex_palette = {
    "M": "#BFBFBF",   # muted blue
    "F": "#4D4D4D" ,    # muted orange
}
cued_uncued_palette = sns.color_palette(['#63B4D1','#B370B0'])

fig, ax = plt.subplots(1,4, figsize=(20,5))
#perf per order
res_order = ttest_ind(data.loc[data.order=='Cued first','accuracy'],data.loc[data.order=='Uncued first','accuracy'])
sns.kdeplot(data=data, x='accuracy',hue='order', ax=ax[0], palette=cued_uncued_palette,
            fill=True, alpha=.5)
how_many_order = len(data.loc[(data.order=='Cued first') & (data['p-value']>=0.05)])

#perf per sex
res_sex = ttest_ind(data.loc[data.sex=='F','accuracy'],data.loc[data.sex=='M','accuracy'],
                       equal_var=False)
sns.kdeplot(data=data, x='accuracy',hue='sex', ax=ax[1], palette=sex_palette,
            fill=True, alpha=.5)
how_many_sex = len(data.loc[(data.sex=='F') & (data['p-value']>=0.05)])

res_hand = ttest_ind(data.loc[data.handedness=='Left','accuracy'],data.loc[data.handedness=='Right','accuracy'],
                       equal_var=False)
# sns.kdeplot(data=data, x='accuracy',hue='handedness', ax=ax[2])
how_many_hand = len(data.loc[(data.handedness=='Left') & (data['p-value']>=0.05)])

#for handedness, do a permutation test
n_iter = 1000

lefties = data.loc[data.handedness=='Left', 'accuracy']
all_accuracies = data.accuracy

perm_means = []
for _ in range(n_iter):
    sample = np.random.choice(all_accuracies, size=sum(left_handed), replace=False)
    perm_means.append(np.mean(sample))

# p_value = np.mean(np.abs(perm_means - np.mean(all_accuracies)) >= 
#                   np.abs(np.mean(lefties) - np.mean(all_accuracies)))

p_value = (np.sum(np.abs(perm_means) >= np.abs(np.mean(lefties))) + 1) / (len(perm_means) + 1)

sns.kdeplot(x=perm_means, ax=ax[2], color="#FFAE70", fill=True, alpha=.5, 
            label="Null distribution")
ax[2].axvline(np.mean(lefties), color="#F96900", linewidth=3)
#nice combo: "#6BA292" and "#1B5C49", AF7595 and 8C2155
ax[2].text(
    np.mean(lefties),
    ax[2].get_ylim()[1] * 0.9,  # near top of plot
    f"Average (left-handed) = {np.mean(lefties):.2f}",
    rotation=90,
    color="k",
    va="top",
    ha="right"
)

#for age, do a regression
res_age = pearsonr(data['age'], data['accuracy'])
sns.regplot(data=data, x='age', y='accuracy', label=f'r={round(res_age.statistic, ndigits=3)}, p={round(res_age.pvalue, ndigits=3)}',
            color='k')
ax[2].legend()
ax[2].set_xlabel('accuracy')
ax[3].legend()
sns.despine(fig=fig, trim=True)

# fig.savefig('./figures/subject_differences.svg', bbox_inches='tight')
# fig.savefig('./figures/subject_differences.png', bbox_inches='tight')
# and improve esthetic aspect
# %%
