#%% imports
# general
import pandas as pd 
import numpy as np
import mne
import ast
from preprocessing_function import preprocessing, get_rt, get_stim, get_response
from warnings import filterwarnings
from mne import set_log_level as mne_log

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from mne.decoding import Vectorizer
from mne.preprocessing import Xdawn

# plotting
import seaborn as sns 
import matplotlib.pyplot as plt

#%% data specifications
subjects=['pilot3','136','358','916','347','766','661','959','205','400',
          '756','897','420','804','207','196','295','402','722','720',
          '966','568','482','307','730','228','785','640','457','183',
          '759','278','732','441','930','755','689','877','312','751',
          '443','223']
nblocks_calib=4
nblocks_test=3
event_id=[1,2]
stim_events=[11,12]
relevant_id=[11,12,1001,1002,1003]

figpath='./presentations/figures/EEG paper/'
data_sc = []
data_perm=[]
n_components=8
tmin, tmax = 0.5, 0.9

col_list=['Subject', 'Condition', 'Trial', 'Cue', 'Stimulus', 'Response', 'RT',
          'Estimated class','Proba 0', 'Proba 1']

filterwarnings(action='ignore', category=UserWarning)
mne_log('CRITICAL')

val_proba=np.zeros((len(subjects),420))
all_cues=np.zeros((len(subjects), 240))
all_stim=np.zeros((len(subjects), 240))
all_resp=np.zeros((len(subjects), 240))
all_rt=np.zeros((len(subjects), 240))
all_sub=np.zeros((len(subjects),240), dtype='U3')
all_trials=np.zeros((len(subjects),240))

#%% model specifications, general
# best_model = pd.read_csv('./best_model.csv')
# best_model['freq'] = best_model['freq'].apply(ast.literal_eval)
fmin, fmax = 4,8
scoring = 'accuracy'

components = 5
#%% import data, train model and test on uncued
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
    
    # (fmin,fmax) = best_model[best_model.subject==s].freq.to_numpy()[0]

    filtered = rw.copy().filter(fmin, fmax, fir_design='firwin', phase='minimum') 
    epochs = mne.Epochs(filtered, stim_events, event_id, tmin, tmax,
                            proj=False, baseline=None, metadata=metadata,#detrend=1, 
                            preload=True, event_repeated='merge')
    evt_sorted=np.sort(epochs.events.view('int,int,int'), order=['f1'], axis=0).view(int)
    labels = evt_sorted[:, -1]
    X = epochs#.get_data() #np.concatenate((ep_subj.get_data(), eptest_subj.get_data()))
    y = labels

    epochs.events[:, 2] = y
    epochs.event_id = {'anticip_face': 1, 'anticip_sound': 2}  
    # val_proba = np.zeros(y_train.shape)
    clf = make_pipeline(
        Xdawn(n_components=components),
        Vectorizer(),
        MinMaxScaler(),
        LogisticRegression(penalty="l1", solver="liblinear"),
        )
    clf.fit(X, y)
    
    train_proba=clf.predict_proba(X)
    rt_cued=get_rt(filtered)
    resp_cued=get_response(filtered)
    pred_class_train=np.argmax(train_proba, axis=1)
    sub_list=[s]*len(rt_cued)
    cond=['Cued']*len(rt_cued)
    trial_ix=[i for i in range (len(rt_cued))]
    stimuli=get_stim(filtered)
    df_train=pd.DataFrame(np.array([sub_list, cond, trial_ix, labels, np.squeeze(stimuli.T)+1, 
                          resp_cued, rt_cued, clf.classes_[pred_class_train], train_proba[:,0],train_proba[:,1]]).T, 
                         columns=col_list)
    
    if len(data_sc)==0:
        data_score = df_train
    else:
        data_score = pd.concat([data_score, df_train])
        
    
    # now handle uncued data 
    datapath='C:/Users/madln/Documents/data/expe1/{}/{}_test_000{}'.format(s,{},{}) #'E:\\PhD\\my_data\\{}\\{}_calib_000{}'.format(s,{},{})
    raws_test=[]
    for b in range(nblocks_test):
        filename = datapath.format(s, b + 1)
        intraw = mne.io.read_raw_brainvision(filename + '.vhdr', preload=True)
        raws_test.append(intraw)

    rw_test = mne.concatenate_raws(raws_test)
    events_test, _ = mne.events_from_annotations(rw_test)
    event_ix = np.isin(events_test[:, 2], [13])
    stim_events = events_test[event_ix]

    subevent_ix = np.isin(events_test[:, 2], relevant_id)
    subevents = events_test[subevent_ix]
    metadata = pd.DataFrame({
        'event_time': stim_events[:, 0],
        'trial_number': np.arange(len(stim_events)),
        'label': stim_events[:, 2]
    })
    filtered = rw_test.copy().filter(fmin, fmax, fir_design='firwin', phase='minimum') 
    eptest = mne.Epochs(filtered, stim_events, [13], tmin, tmax,
                            proj=False, baseline=None, metadata=metadata,#detrend=1, 
                            preload=True, event_repeated='merge')
    # eptest, y_test, _, eegtest=my_preprocessing3(datapath_test.format(s,{},{}),s,nblocks_test,[13],relevant_id,-1,2,cut_low=fmin,cut_high=fmax)

    X_test = eptest
    
    test_proba=clf.predict_proba(X_test)
    rt_uncued=get_rt(filtered)
    resp_uncued=get_response(filtered)
    pred_class_test=np.argmax(test_proba, axis=1)
    sub_list=[s]*len(rt_uncued)
    cond=['Uncued']*len(rt_uncued)
    trial_ix=[i for i in range (len(rt_uncued))]
    stimuli=get_stim(filtered)
    df_test=pd.DataFrame(np.array([sub_list, cond, trial_ix, [0]*len(sub_list), np.squeeze(stimuli.T)+1, 
                          resp_uncued, rt_uncued, clf.classes_[pred_class_test], test_proba[:,0],test_proba[:,1]]).T, 
                         columns=col_list)
    
    if len(data_sc)==0:
        data_sc = df_test
    else:
        data_sc = pd.concat([data_sc, df_test])
        
# data_sc.to_csv('uncued_classification_and_behavior_4-8Hz.csv')
data_score.to_csv('cued_classification_and_behavior_4-8Hz.csv')