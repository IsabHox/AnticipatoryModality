import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from preprocessing_function import preprocessing, get_rt, get_stim, get_response
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import pandas as pd
import seaborn as sns

from mne.decoding import Vectorizer
from mne.preprocessing import Xdawn

from warnings import filterwarnings
from mne import set_log_level as mne_log
import mne

import ast

from sklearn.base import clone

#%% data specifications
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

figpath='./presentations/figures/EEG paper/'
data_sc = []
data_perm=[]
n_components=8
tmin, tmax = 0.5, 0.9

best_model = pd.read_csv('./best_model.csv')
best_model['freq'] = best_model['freq'].apply(ast.literal_eval)

fmin,fmax = 4,8 

scoring = 'accuracy'

n_perm = 1000

filterwarnings(action='ignore', category=UserWarning)
mne_log('CRITICAL')

val_proba=np.zeros((len(subjects),420))
all_cues=np.zeros((len(subjects), 240))
all_stim=np.zeros((len(subjects), 240))
all_resp=np.zeros((len(subjects), 240))
all_rt=np.zeros((len(subjects), 240))
all_sub=np.zeros((len(subjects),240), dtype='U3')
all_trials=np.zeros((len(subjects),240))

#%% pipeline and gridsearch specifications
pipeline = Pipeline(
    [
        ("reduce_dim",  "passthrough"),    
        ("scaling", MinMaxScaler()),
        ("classify", LogisticRegression(penalty="l1", solver="liblinear")),
    ])

#N_FEATURES_OPTIONS = [4]#, 4, 8]
#C_OPTIONS = [1, 10, 100, 1000]
#param_grid = [
#    {   "reduce_dim":[Xdawn()],
#        "reduce_dim__n_components": N_FEATURES_OPTIONS,
#    }
#]
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
keys= [0,10,42]
key=0
ttsplit = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=key)
#grid = GridSearchCV(pipeline, n_jobs=-2, cv = cv, param_grid=param_grid)

#%% on to a fitting
components = [5]

data_perm=[]    
for subject in range(len(subjects)):
    s=subjects[subject]
    print(s)
    datapath='C:/Users/madln/Documents/data/expe1/{}/{}_calib_000{}'.format(s,{},{}) #'E:\\PhD\\my_data\\{}\\{}_calib_000{}'.format(s,{},{})
    raws=[]
    for b in range(nblocks_calib):
        filename = datapath.format(s, b + 1)
        intraw = mne.io.read_raw_brainvision(filename + '.vhdr', preload=True)
        raws.append(intraw)

    rw = mne.concatenate_raws(raws)
    events, _ = mne.events_from_annotations(rw)
    event_ix = np.isin(events[:, 2], event_id)
    stim_events = events[event_ix]

    subevent_ix = np.isin(events[:, 2], relevant_id)
    subevents = events[subevent_ix]
    metadata = pd.DataFrame({
        'event_time': stim_events[:, 0],
        'trial_number': np.arange(len(stim_events)),
        'label': stim_events[:, 2]
    })
    raw = mne.concatenate_raws(raws)
    # (fmin,fmax) = best_model[best_model.subject==s].freq.to_numpy()[0]

    filtered = raw.copy().filter(fmin, fmax, fir_design='firwin', phase='minimum') 
    epochs = mne.Epochs(filtered, stim_events, event_id, tmin, tmax,
                            proj=False, baseline=None, metadata=metadata,#detrend=1, 
                            preload=True, event_repeated='merge')
    evt_sorted=np.sort(epochs.events.view('int,int,int'), order=['f1'], axis=0).view(int)
    labels = evt_sorted[:, -1]
    X = epochs#.get_data() #np.concatenate((ep_subj.get_data(), eptest_subj.get_data()))
    y = labels
    
    for i, (trainix, testix) in enumerate(ttsplit.split(X, y)):
        X_train, y_train = X[trainix], y[trainix]
        X_test, y_test = X[testix], y[testix]

    epochs.events[:, 2] = y
    epochs.event_id = {'anticip_face': 1, 'anticip_sound': 2}  
    val_proba = np.zeros(y_train.shape)
    clf = make_pipeline(
        Xdawn(n_components=5),
        Vectorizer(),
        MinMaxScaler(),
        LogisticRegression(penalty="l1", solver="liblinear"),
        )
    clf.fit(X_train, y_train)
    val_proba = clf.predict(X_test)
    sp_mdm = accuracy_score(y_test, val_proba)
    
    scores_perm = []

    for i in range(n_perm):
        clf2 = clone(clf)
        y_train_perm = np.random.permutation(y_train)
        clf2.fit(X_train, y_train_perm)
        scores_perm.append(clf2.score(X_test, y_test))

    # p-value
    p_value = (np.sum(np.array(scores_perm) >= sp_mdm) + 1) / (n_perm + 1)
    
    data_perm.append({'subject':s, 'freq':(fmin, fmax),'components':5, 'time':(tmin, tmax),
                        'classif': 'XDawn+LogReg', 'accuracy':sp_mdm, 'chance':np.percentile(scores_perm, 95),
                        'chance (average)':np.mean(scores_perm),'p-value':p_value})
    
    # Normalized confusion matrix
    # cm = confusion_matrix(y_test, val_proba)
    # cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    # # Plot confusion matrix
    # fig, ax = plt.subplots(figsize=(5.5,5), layout="constrained")
    # sns.heatmap(cm_normalized, annot=True,  fmt='.2%', cmap='Blues',ax=ax)
    # ax.set(title=f"Normalized Confusion matrix, participant {s}")
    # tick_marks = np.arange(len(epochs.event_id))+0.5
    # plt.xticks(tick_marks, epochs.event_id)
    # plt.yticks(tick_marks, epochs.event_id)
    # ax.set(ylabel="True label", xlabel="Predicted label")
    # fig.savefig(f'./figures/confusion_matrix_test_{s}.svg')

    # plot topomaps of Xdawn learned features
    # n_filter = 5
    # fig, axes = plt.subplots(
    #     nrows=len(event_id),
    #     ncols=n_filter,
    #     figsize=(n_filter, len(event_id) * 2),
    #     layout="constrained",
    # )
    # fitted_xdawn = clf.steps[0][1]
    # info = mne.create_info(epochs.ch_names, 1, epochs.get_channel_types())
    # info.set_montage(epochs.get_montage())
    # for ii, cur_class in enumerate(sorted(event_id)):
    #     cur_patterns = fitted_xdawn.patterns_[str(cur_class)]
    #     pattern_evoked = mne.EvokedArray(cur_patterns[:n_filter].T, info, tmin=0)
    #     pattern_evoked.plot_topomap(
    #         times=np.arange(n_filter),
    #         time_format="Component %d" if ii == 0 else "",
    #         colorbar=False,
    #         show_names=False,
    #         axes=axes[ii],
    #         show=False,
    #     )
    #     axes[ii, 0].set(ylabel=cur_class)
    # fig.savefig(f'./figures/xdawn_components_{s}.svg')




# %%
df_perm2 = pd.DataFrame(data_perm)
df_perm2.to_csv(f'./test_set_performance_4-8Hz_{key}.csv')
# %%
