import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from preprocessing_function import preprocessing, get_rt, get_stim, get_response
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import pandas as pd

from mne.decoding import Vectorizer, CSP
from mne.preprocessing import Xdawn

from warnings import filterwarnings
from mne import set_log_level as mne_log
import mne

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
#tmin_bl, tmax_bl=-0.4,0
fmin, fmax = 1, 35
scoring = 'accuracy'
freqbands=[#(None,4),(None,8),(None,13),(None, 35),
           (1,4),(1,8),(1,13),(None,35),
           (4,8),(8,13)]#,(13,20),(20,35)]

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
ttsplit = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
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
    for (fmin, fmax) in freqbands:

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
        for c in components:
            val_proba = np.zeros(y_train.shape)
            cv_mdm = cv.split(X_train, y_train)
            for i, (train, val) in enumerate(cv_mdm):
                clf = make_pipeline(
                    Xdawn(n_components=c),
                    # CSP(n_components=c, reg=None,log=True),
                    Vectorizer(),
                    MinMaxScaler(),
                    # StandardScaler(),
                    #XdawnCovariances(nfilter=8, estimator='oas'),
                    #MDM(metric='riemann'),
                    LogisticRegression(penalty="l1", solver="liblinear"),
                    )
                clf.fit(X_train[train], y_train[train])
                val_proba[val]=clf.predict(X_train[val])
            sp_mdm = accuracy_score(y_train, val_proba)
            data_perm.append({'subject':s, 'freq':(fmin, fmax),'components':c, 'time':(tmin, tmax),
                                'classif': 'MDM', 'accuracy':sp_mdm})

        
df_perm2 = pd.DataFrame(data_perm)
print(df_perm2)

#%% get the best model for each participant
best_per_subject = df_perm2.loc[df_perm2.groupby('subject')['accuracy'].idxmax()]

# %% save those results
best_per_subject.to_csv('./best_model.csv')


# %%
