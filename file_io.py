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
from dataloader import load_dataset
from torch.utils import data
import torch


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