
from glob2 import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from skimage import exposure
import traceback
import sklearn
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tsai.all import (get_UCR_data, combine_split_data, TSDatasets, Categorize,ClassificationInterpretation,
                      TSDataLoaders,TSStandardize, InceptionTime, Learner, accuracy, load_learner_all)


def normalize99(X):

    sklearn.preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
        
    return X

def rescale01(x):
        
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
        
    return x


def preprocess_data(x):
    
    x = normalize99(x)
    x = rescale01(x)
    
    x = list(x)
    
    return x


def split_list(data, chunk_size = 200):

    split_data = [] 
    
    for dat in data:
        
        dat_split = np.split(dat,range(0, len(dat), 200), axis=0)
        
        dat_split = [list(x) for x in dat_split if len(x) == 200]
        
        split_data.extend(dat_split)
        
    return split_data


def read_gapseq_data(file_paths, X, y, file_names, label = 0):
    

    for file_path in file_paths:
        
        try:
            
            with open(file_path) as f:
                
                d = json.load(f)
            
                data = np.array(d["data"])
                
                data = [dat for dat in data]
                
                # data = split_list(data, chunk_size = 200)
                
                for dat in data:
                    
                    if len(dat) > 200:
                        
                        dat = dat[:1200]
                    
                        file_name = [os.path.basename(file_path)]*len(dat)
                        
                        dat = preprocess_data(dat)
                        
                        X.append(list(dat))
                        y.append(label)
                        file_names.append(file_name)
            
        except:
            # print(traceback.format_exc())
            pass
        
    return X, y, file_names


            



# directory_path = r"/run/user/26623/gvfs/smb-share:server=physics.ox.ac.uk,share=dfs/DAQ/CondensedMatterGroups/AKGroup/Jagadish/Traces for ML"


# complimentary_files_path = os.path.join(directory_path, "Complementary Traces")
# noncomplimentary_files_path = os.path.join(directory_path, "Non_Complementary Traces")


# complimentary_files = glob(complimentary_files_path + "*/*_gapseqML.txt")
# noncomplimentary_files = glob(noncomplimentary_files_path + "*/*_gapseqML.txt")





# X = []
# y = []
# file_names = []

# X, y, file_names = read_gapseq_data(complimentary_files, X, y, file_names, 0)
# X, y, file_names = read_gapseq_data(noncomplimentary_files, X, y, file_names, 1)





# with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([X, y, file_names], f)



with open('objs.pkl','rb') as f:  # Python 3: open(..., 'rb')
    X, y, file_names = pickle.load(f)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# X, y, file_names = shuffle(X, y, file_names)  

# X_train, X_test, y_train, y_test = train_test_split(np.array(X),
#                                                     np.array(y),
#                                                     train_size=0.8,
#                                                     shuffle=True)

# X_train = np.expand_dims(X_train,1)
# X_test = np.expand_dims(X_test,1)

# X, y, splits = combine_split_data([np.array(X_train), np.array(X_test)], [np.array(y_train), np.array(y_test)])


# tfms  = [None, [Categorize()]]
# dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)

# dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[1024,128], batch_tfms=[TSStandardize()], num_workers=0)

    
# # X,y = next(iter(dls.valid))
# # dls.show_batch(sharey=True)

        

# model = InceptionTime(dls.vars, dls.c)
# learn = Learner(dls, model, metrics=accuracy)
# # learn.save('stage0')

# # learn.load('stage0')
# # learn.lr_find()

# learn.fit_one_cycle(25, lr_max=1e-4)
# learn.save('stage1')

# learn.recorder.plot_metrics()

# learn.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')



learn = load_learner_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')
dls = learn.dls
valid_dl = dls.valid
    
valid_probas, valid_targets, valid_preds = learn.get_preds(dl=valid_dl, with_decoded=True)
accuracy = (valid_targets == valid_preds).float().mean().numpy()

learn.show_probas()
     
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
   

    

learn.show_results()
     

    
    
    
    
    
    
    
    
    
    