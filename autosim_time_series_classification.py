# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 08:43:58 2022

@author: turnerp
"""


import pandas
import mat4py
from glob2 import glob
import os
import numpy as np
import random

from sklearn.model_selection import train_test_split

from tsai.all import (get_UCR_data, combine_split_data, TSDatasets, Categorize,
                      TSDataLoaders,TSStandardize, InceptionTime, Learner, accuracy,
                      load_learner_all, ClassificationInterpretation)




def get_autosim_data(shuffle=True):

    mat_paths = glob(r"C:\Users\turnerp\PycharmProjects\GAPSEQML\AutoSIM\scripts_SiMREPS\data\*.mat")
    
    wt_data = [path for path in mat_paths if "WT" in os.path.basename(path)][:1]
    mut_data = [path for path in mat_paths if "MUT" in os.path.basename(path)][:1]
    
    wt_data = [dat for path in wt_data for dat in mat4py.loadmat(path)["traces"]]
    mut_data = [dat for path in mut_data for dat in mat4py.loadmat(path)["traces"]]
    
    wt_labels = [0]*len(wt_data)
    mut_labels = [1]*len(mut_data)
    
    X = np.concatenate((wt_data,mut_data))
    y = np.concatenate((wt_labels,mut_labels))
     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle = shuffle)
    
    X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])
    
    return X, y, splits


X, y, splits = get_autosim_data()






# tfms  = [None, [Categorize()]]
# dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)

# train = dsets.train



# dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64], batch_tfms=[TSStandardize()], num_workers=0)


# # dls.show_batch(sharey=True)

# model = InceptionTime(dls.vars, dls.c)
# learn = Learner(dls, model, metrics=accuracy)
# # learn.save('stage0')

# # learn.load('stage0')
# # learn.lr_find()

# learn.fit_one_cycle(25, lr_max=1e-3)
# learn.save('stage1')

# learn.recorder.plot_metrics()

# learn.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')






#inference

# learn = load_learner_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')
# dls = learn.dls
# valid_dl = dls.valid


# valid_probas, valid_targets, valid_preds = learn.get_preds(dl=valid_dl, with_decoded=True)

# valid_accuracy = (valid_targets == valid_preds).float().mean().numpy()

# learn.show_probas()

# interp = ClassificationInterpretation.from_learner(learn)
# interp.plot_confusion_matrix()





















