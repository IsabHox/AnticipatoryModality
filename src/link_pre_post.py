from preprocessing_function import preprocessing
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from mne.stats import spatio_temporal_cluster_1samp_test


#%% so that mne doesn't print it all
mne.set_log_level('ERROR')

#%% data import
# subjects=['897']
subjects=['pilot3','136','358','916','347','766','661','959','205','400',
          '756','897','420','804','207','196','295','402','722','720',
          '966','568','482','307','730','228','785','640','457','183',
          '759','278','732','441','930','755','689','877','312','751',
          '443','223']
#new participants since: 
# subjects = ['pilot3','136','358','897']#]
nblocks_calib=4
nblocks_test=3
event_id=[1,2]
stim_events=[11,12]
relevant_id=[11,12,1001,1002,1003]
tmin=-1
tmax=2

#%% where processing is cached
cache_dir_pre = Path("../../data/expe1/cache_no_reref")
cache_dir_pre.mkdir(exist_ok=True)

# %% lets see now if we do single trial regressions
from scipy.stats import pearsonr, linregress

prestims_1 = []
poststims_1 = []
prestims_2 = []
poststims_2 = []

slopes = np.zeros((len(subjects),))
intercepts = np.zeros((len(subjects),))
rs = np.zeros((len(subjects),))
ps = np.zeros((len(subjects),))
ses = np.zeros((len(subjects),))

dpr = []
dps = []

for i,subject in enumerate(subjects):
    print(f"Processing subject {subject}")
    cache_file_pre = cache_dir_pre / f"{subject}-evoked_pre_post.npz"
    
    if cache_file_pre.exists():
        print("Loading cached pre-stimulus evoked")
        data = np.load(cache_file_pre)
        prestims_1.append(data['prestim_1']),
        poststims_1.append(data['poststim_1']),
        prestims_2.append(data['prestim_2']),
        poststims_2.append(data['poststim_2']),
        if i==0:
            datapath= '../../data/expe1/{}/{}_calib_000{}'.format(subject,{},{})
            epochs, labels, _, _,eegdata=preprocessing(datapath,subject,nblocks_calib,event_id,relevant_id,tmin,tmax,cut_low=None,cut_high=35., reref = False) #previously: 8., no cut_low // previously: cut_high=35.
        # continue
    else:
        datapath= '../../data/expe1/{}/{}_calib_000{}'.format(subject,{},{})
        
        # filtering: 0.1-35 for all, 8-13 for alpha band specifically
        epochs, labels, _, _,eegdata=preprocessing(datapath,subject,nblocks_calib,event_id,relevant_id,tmin,tmax,cut_low=None,cut_high=35., reref = False) #previously: 8., no cut_low // previously: cut_high=35.

        poststim_epochs = epochs.copy().crop(0.9,1.4) #.crop()
        prestim_epochs = epochs.copy().crop(0,0.9) #.
        
        pr1 = prestim_epochs['1'].average().data
        pr2 = prestim_epochs['2'].average().data
        ps1 = poststim_epochs['1'].average().data
        ps2 = poststim_epochs['2'].average().data
        
        prestims_1.append(pr1)
        poststims_1.append(ps1)
        prestims_2.append(pr2)
        poststims_2.append(ps2)
            
        np.savez(
            cache_file_pre,
            prestim_1=pr1,
            poststim_1=ps1,
            prestim_2=pr2,
            poststim_2=ps2,
        )
        
    pr1 = prestims_1[i]
    pr2 = prestims_2[i]
    ps1 = poststims_1[i]
    ps2 = poststims_2[i]
    #then, one regression per subject
    delta_pre = np.mean(np.abs(mne.combine_evoked(
                [mne.EvokedArray(pr1,epochs.info),mne.EvokedArray(pr2,epochs.info)], 
                weights=[1, -1]).data), axis=1)
    delta_post = np.mean(np.abs(mne.combine_evoked(
                [mne.EvokedArray(ps1,epochs.info),mne.EvokedArray(ps2,epochs.info)], 
                weights=[1, -1]).data),axis=1)
    
    slopes[i], intercepts[i], rs[i], ps[i], ses[i] = linregress(delta_pre, delta_post)
    
    dpr.append(delta_pre)
    dps.append(delta_post)
    
# prestims=np.swapaxes(prestims, 1, 2)
# poststims=np.swapaxes(poststims, 1, 2)

# %% make some plots
delta_pre_all = np.array(dpr)
delta_post_all = np.array(dps)

mean_pre = delta_pre_all.mean(axis=0)
mean_post = delta_post_all.mean(axis=0)

sem_pre = delta_pre_all.std(axis=0) / np.sqrt(delta_pre_all.shape[0])
sem_post = delta_post_all.std(axis=0) / np.sqrt(delta_post_all.shape[0])

# pos = mne.find_layout(epochs.info).pos

layout = mne.find_layout(epochs.info)
pos = layout.pos[:, :2]

x = pos[:, 0]
y = pos[:, 1]
x_norm = (x - x.min()) / (x.max() - x.min())
y_norm = (y - y.min()) / (y.max() - y.min())

import matplotlib.colors as mcolors

x = pos[:, 0]
y = pos[:, 1]
x_norm = (x - x.min()) / (x.max() - x.min())
y_norm = (y - y.min()) / (y.max() - y.min())

Rgd = x_norm
Ggd = np.zeros_like(x_norm)
Bgd = 1 - x_norm

colors_gd = np.stack([Rgd, Ggd, Bgd], axis=1)


df = pd.DataFrame({
    "pre":np.squeeze(delta_pre_all.reshape([-1,1], order='C')),
    "post":np.squeeze(delta_post_all.reshape([-1,1], order='C')),
    "electrode":epochs.info['ch_names']*len(subjects),
    "subject":np.repeat(subjects,len(epochs.info['ch_names']))
})


fig, ax = plt.subplots(1,4, figsize=(20,5))
ax1, ax2, ax3, ax4 = ax

# Share y-axis only for first three
ax2.sharey(ax1)
ax3.sharey(ax1)

# Optional: hide redundant y tick labels
# ax2.tick_params(labelleft=False)
# ax3.tick_params(labelleft=False)

sns.scatterplot(data=df, x="pre", y="post", ax=ax[1], hue='electrode', palette=colors_gd,
                legend=False)

ax[0].set_xlabel('Pre-stimulus |ΔERP| (V)')
ax[0].set_ylabel('Post-stimulus |ΔERP| (V)')

sns.despine(ax=ax[0], offset=10)
ax[1].set_xlabel('Pre-stimulus |ΔERP| (V)')
ax[1].set_ylabel('')

sns.despine(ax=ax[1], offset=10)

# Rhb = x_norm
# Ghb = np.zeros_like(x_norm)
# Bhb = 1 - x_norm
# colors_hb = np.stack([Rgd, Ggd, Bgd], axis=1)
import matplotlib.colors as mcolors

# c_left = np.array(mcolors.to_rgb("#7AB"))
# c_right = np.array(mcolors.to_rgb("#EDA"))
[c_left, c_right] = sns.color_palette("Set2", n_colors=2)
colors_hb = (1 - y_norm)[:, None] * c_left + y_norm[:, None] * c_right

sns.scatterplot(data=df, x="pre", y="post", ax=ax[0], hue='electrode', palette=colors_hb,
                legend=False)

# add little head for legend
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
ax_topo = inset_axes(ax[0], width="20%", height="20%",loc="upper left")
ax_topo.set_aspect("equal")

layout = mne.find_layout(epochs.info)
pos2d = layout.pos[:, :2]

pos2d = pos2d - pos2d.min(axis=0)
pos2d = pos2d / pos2d.max(axis=0)

ax_topo.scatter(pos2d[:, 0], pos2d[:, 1], c=colors_hb, s=20)

ax_topo.set_xticks([])
ax_topo.set_yticks([])
# ax_topo.set_title("Electrode layout", fontsize=8)

# optional: draw head circle
# circle = plt.Circle((0.5, 0.5), 0.5, transform=ax_topo.transAxes,
#                     fill=False, color='black', linewidth=1)
circle = plt.Circle((0.5, 0.5), 0.5, fill=False, color='black', linewidth=1)
ax_topo.add_patch(circle)

nose = np.array([   # tip
    [0.45, 1],
    [0.5, 1.1],
    [0.55, 1]
])

ax_topo.plot(nose[:, 0], nose[:, 1], color='black', linewidth=1)

from matplotlib.patches import Ellipse

left_ear = Ellipse((0.0, 0.5), width=0.1, height=0.3,
                   angle=0, fill=False, color='black', linewidth=1)

right_ear = Ellipse((1.0, 0.5), width=0.1, height=0.3,
                    angle=0, fill=False, color='black', linewidth=1)

ax_topo.add_patch(left_ear)
ax_topo.add_patch(right_ear)

sns.despine(ax=ax_topo, bottom=True, left=True)


ax_topo2 = inset_axes(ax[1], width="20%", height="20%",loc="upper left")
ax_topo2.set_aspect("equal")
ax_topo2.scatter(pos2d[:, 0], pos2d[:, 1], c=colors_gd, s=20)

ax_topo2.set_xticks([])
ax_topo2.set_yticks([])
circle = plt.Circle((0.5, 0.5), 0.5, fill=False, color='black', linewidth=1)
ax_topo2.add_patch(circle)


ax_topo2.plot(nose[:, 0], nose[:, 1], color='black', linewidth=1)
left_ear = Ellipse((0.0, 0.5), width=0.1, height=0.3,
                   angle=0, fill=False, color='black', linewidth=1)

right_ear = Ellipse((1.0, 0.5), width=0.1, height=0.3,
                    angle=0, fill=False, color='black', linewidth=1)
ax_topo2.add_patch(left_ear)
ax_topo2.add_patch(right_ear)

sns.despine(ax=ax_topo2, bottom=True, left=True)



# and then in the second axis plot the regression coefficients
sns.histplot(x=[r**2 for r in rs], ax=ax[3], color='grey')
ax[3].set_xlabel('R²')
sns.despine(ax=ax[3], offset=10)

# and then try out the regression per participant

for s in subjects:
    sns.regplot(data=df[df['subject']==s], x='pre', y='post',scatter=False, ax=ax[2], color='grey')
sns.despine(ax=ax[2], offset=10)
ax[2].set_xlabel('Pre-stimulus |ΔERP| (V)')
ax[2].set_ylabel('')

# fig.savefig('./figures/electrode_regression.svg')
# %% could also do one plot per subject:
# for s in range(len(subjects)):
    
#     mean_pre = dpr[s]
#     mean_post = dps[s]
    
#     df = pd.DataFrame({
#         "pre": mean_pre,
#         "post": mean_post,
#         "electrode":epochs.info['ch_names']
#     })
#     fig, ax = plt.subplots()
#     sns.scatterplot(data=df, x="pre", y="post", ax=ax, hue='electrode', palette=colors,
#                     legend=False)

#     ax.set_xlabel('Pre-stimulus |ΔERP| (V)')
#     ax.set_ylabel('Post-stimulus |ΔERP| (V)')
#     ax.set_title(f'Electrode-wise relationship, subject {subjects[s]}')

#     sns.despine(ax=ax, offset=10)

#     # add little head for legend
#     from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#     ax_topo = inset_axes(ax, width="20%", height="20%",loc="upper left")
#     ax_topo.set_aspect("equal")

#     layout = mne.find_layout(epochs.info)
#     pos2d = layout.pos[:, :2]

#     pos2d = pos2d - pos2d.min(axis=0)
#     pos2d = pos2d / pos2d.max(axis=0)

#     ax_topo.scatter(pos2d[:, 0], pos2d[:, 1], c=colors, s=50)

#     ax_topo.set_xticks([])
#     ax_topo.set_yticks([])
#     circle = plt.Circle((0.5, 0.5), 0.5, fill=False, color='black', linewidth=1)
#     ax_topo.add_patch(circle)

#     nose = np.array([   # tip
#         [0.47, 1],
#         [0.5, 1.05],
#         [0.53, 1]
#     ])

#     ax_topo.plot(nose[:, 0], nose[:, 1], color='black', linewidth=1)

#     from matplotlib.patches import Ellipse

#     left_ear = Ellipse((0.0, 0.5), width=0.1, height=0.3,
#                     angle=0, fill=False, color='black', linewidth=1)

#     right_ear = Ellipse((1.0, 0.5), width=0.1, height=0.3,
#                         angle=0, fill=False, color='black', linewidth=1)

#     ax_topo.add_patch(left_ear)
#     ax_topo.add_patch(right_ear)

#     sns.despine(ax=ax_topo, bottom=True, left=True)

# %% link to performance
perf_data = pd.read_csv('./test_set_performance_4-8Hz_0.csv')
fig, ax = plt.subplots()
res = pearsonr(rs, perf_data['accuracy'])
sns.regplot(x=perf_data['accuracy'], y=rs, ax=ax)
ax.set_ylim(0,1)
# %%
