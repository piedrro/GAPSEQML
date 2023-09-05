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
import random


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


def read_gapseq_data(file_paths, X  = [], y = [], file_names = [], label = 0, trace_limit = 1200):
    

    for file_path in file_paths:
        
        try:
            
            with open(file_path) as f:
                
                d = json.load(f)
            
                data = np.array(d["data"])
                
                data = [dat for dat in data]
                
                # data = split_list(data, chunk_size = 200)
                
                for dat in data:
                    
                    if len(dat) > 200:
                        
                        dat = dat[:trace_limit]
                    
                        file_name = os.path.basename(file_path)
                        
                        dat = preprocess_data(dat)
                        
                        X.append(list(dat))
                        y.append(label)
                        file_names.append(file_name)
            
        except:
            # print(traceback.format_exc())
            pass
        
    return X, y, file_names






def shuffle_train_data(train_data):
      
    dict_names = list(train_data.keys())     
    dict_values = list(zip(*[value for key,value in train_data.items()]))
    
    random.shuffle(dict_values)
    
    dict_values = list(zip(*dict_values))
    
    train_data = {key:list(dict_values[index]) for index,key in enumerate(train_data.keys())}
    
    return train_data
                    

def limit_train_data(train_data, num_files):
    
    for key,value in train_data.items():
        
        train_data[key] = value[:num_files]
        
    return train_data

def split_dataset(X,y,file_names,ratio_train,val_test_split):
    

    dataset = {"X":np.array(X),"y":np.array(y),"file_names":np.array(file_names)}
    
    train_dataset = {"X":[],"y":[],"file_names":[]}
    validation_dataset = {"X":[],"y":[],"file_names":[]}
    test_dataset = {"X":[],"y":[],"file_names":[]}
    
    for label in np.unique(dataset["y"]):
        
        label_file_names = np.unique(np.extract(dataset["y"]==label,dataset["file_names"]))
        
        for file_name in label_file_names:
            
            indices = np.argwhere(dataset["file_names"]==file_name).flatten()
            
            
            
            train_indices, val_indices = train_test_split(indices,
                                                          train_size=ratio_train,
                                                          shuffle=True)
            
            val_indices, test_indices = train_test_split(val_indices,
                                                          train_size=val_test_split,
                                                          shuffle=True)
            
            for key,value in dataset.items():
                
                train_data = dataset[key][train_indices].tolist()
                validation_data = dataset[key][val_indices].tolist()
                test_data = dataset[key][test_indices].tolist()
                
                train_dataset[key].extend(train_data)
                validation_dataset[key].extend(validation_data)
                test_dataset[key].extend(test_data)
            
    train_dataset = shuffle_train_data(train_dataset) 
    validation_dataset = shuffle_train_data(validation_dataset) 
    test_dataset = shuffle_train_data(test_dataset)

    return train_dataset, validation_dataset, test_dataset