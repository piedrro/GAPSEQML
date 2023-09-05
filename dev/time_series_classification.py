# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 08:23:27 2022

@author: turnerp
"""


import matplotlib.pyplot as plt

from tsai.all import (get_UCR_data, combine_split_data, TSDatasets, Categorize,
                      TSDataLoaders,TSStandardize, InceptionTime, Learner, accuracy)
# computer_setup()


def get_data(dsid = 'NATOPS' ):

    X_train, y_train, X_test, y_test  = get_UCR_data(dsid, return_split=True)
    X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])
    
    return X, y, splits


X_train, y_train, X_test, y_test  = get_UCR_data('NATOPS', return_split=True)
X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])




X, y, splits = get_data('ACSF1' )

tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)

dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64,1024], batch_tfms=[TSStandardize()], num_workers=0)


# X,y = next(iter(dls.valid))
# dls.show_batch(sharey=True)

model = InceptionTime(dls.vars, dls.c)
learn = Learner(dls, model, metrics=accuracy)
# learn.save('stage0')

# learn.load('stage0')
# learn.lr_find()

# learn.fit_one_cycle(200, lr_max=1e-3)
# learn.save('stage1')

learn.load('stage1')
learn.recorder.plot_metrics()

# learn.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')

# valid_probas, valid_targets, valid_preds = learn.get_preds(dl=dsets.valid, with_decoded=True)