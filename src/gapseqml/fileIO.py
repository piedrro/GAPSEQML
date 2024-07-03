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
from torch.utils import data
import torch
import random

from gapseqml.dataloader import load_dataset


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

def import_new_ml_data(imported_data, path, label=0, trace_length = 1000,
                       ml_data = {"data":[], "labels":[], "file_names":[]}):
    
    n_traces = 0
    
    file_name = os.path.basename(path)
    
    for i in range(len(imported_data["data"])):
        try:

            data = imported_data["data"][i]
            dataset = imported_data["dataset"][i]
            channels = imported_data["channels"][i]
            
            for dat in data:
                
                try:
                
                    if dat != None:
                    
                        if len(dat) > trace_length:
                            
                            dat = preprocess_data(dat)
                            
                            dat = list(dat)
                            dat = dat[:trace_length]
                        
                            ml_data["data"].append(dat)
                            ml_data["labels"].append(int(label))
                            ml_data["file_names"].append(str(file_name))
                
                            n_traces += 1
                
                except:
                    print(traceback.format_exc())

        except:
            print(traceback.format_exc())
            pass

    return ml_data


def import_legacy_ml_data(imported_data, path, label=0, trace_length=1000,
                          ml_data = {"data":[], "labels":[], "file_names":[]}):

    try:

        n_traces = 0

        file_name = os.path.basename(path)

        traces = imported_data["data"]
        user_labels = imported_data["data_class"]
        nucleotide_labels = imported_data["data_nucleotide"]
        
        for dat in traces:
            
            try:
            
                if dat != None:
            
                    if len(dat) > trace_length:
                        
                        dat = preprocess_data(dat)
                        
                        dat = list(dat)[:trace_length]
                    
                        ml_data["data"].append(dat)
                        ml_data["labels"].append(int(label))
                        ml_data["file_names"].append(str(file_name))
            
                        n_traces += 1
                
            except:
                print(traceback.format_exc())
        
    except:
        print(traceback.format_exc())
        pass

    return ml_data


def import_gapseqml_data(paths, label = 0, trace_length = 1000, 
                         ml_data = {"data":[], "labels":[], "file_names":[]}, ):
    
    if type(paths) != list:
        paths = [paths]

    path_list = []

    for path in paths:
        txt_files = glob(path + "/*.txt")
        json_files = glob(path + "/*.json")
        all_files = txt_files + json_files
        path_list += all_files

    legacy_expected_keys = ["data", "label", "data_class", "data_nucleotide"]
    expected_keys = ["data","states","ml_label","dataset","channels",
                     "user_label","nucleotide_label","import_path"]
    
    if len(path_list) > 0:
        
        for path in path_list:
            
            try:
            
                file, ext = os.path.splitext(path)
                
                if ext == ".txt":
        
                    with open(path) as f:
                        d = json.load(f)
        
                    import_mode = None
                    import_error = False
                    if set(legacy_expected_keys).issubset(d.keys()):
                        import_mode = "legacy"
                    elif set(expected_keys).issubset(d.keys()):
                        import_mode = "new"
                    else:
                        import_error = True
                        
                    if import_error == False:
        
                        if import_mode == "legacy":
                            ml_data = import_legacy_ml_data(d, path, label, trace_length, ml_data)
                            
                        if import_mode == "new":
                            ml_data = import_new_ml_data(d, path, label, trace_length, ml_data)

                if ext == ".json":
                    pass
                
            except:
                print(traceback.format_exc())
            
    if len(ml_data["data"]) > 0:
        print(f"Imported {len(ml_data['data'])} traces with label: {label}")
    else:
        return None
        
    return ml_data



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

def split_datasets(ml_data, ratio_train,val_test_split):
    
    ml_data["data"] = np.array(ml_data["data"])
    ml_data["labels"] = np.array(ml_data["labels"])
    ml_data["file_names"] = np.array(ml_data["file_names"])
    
    train_dataset = {"data":[],"labels":[],"file_names":[]}
    validation_dataset = {"data":[],"labels":[],"file_names":[]}
    test_dataset = {"data":[],"labels":[],"file_names":[]}
    
    for label in np.unique(ml_data["labels"]):
        
        label_file_names = np.unique(np.extract(ml_data["labels"]==label,ml_data["file_names"]))
        
        label_file_names = np.flip(label_file_names)
        
        train_files, test_files = train_test_split(label_file_names,
                                                   train_size=ratio_train,
                                                   shuffle=True)

        test_indices = np.argwhere(np.isin(ml_data["file_names"],test_files)).flatten()
        
        for key,value in ml_data.items():
            
            test_data = ml_data[key][test_indices].tolist()
            test_dataset[key].extend(test_data)
        
        for file_name in train_files:
            
            indices = np.argwhere(ml_data["file_names"]==file_name).flatten()
            
            train_indices, val_indices = train_test_split(indices,
                                                          train_size=val_test_split,
                                                          shuffle=True)
            
            for key,value in ml_data.items():
                
                train_data = ml_data[key][train_indices].tolist()
                validation_data = ml_data[key][val_indices].tolist()
                
                train_dataset[key].extend(train_data)
                validation_dataset[key].extend(validation_data)
                
            
    train_dataset = shuffle_train_data(train_dataset) 
    validation_dataset = shuffle_train_data(validation_dataset) 
    test_dataset = shuffle_train_data(test_dataset)

    return train_dataset, validation_dataset, test_dataset







# def split_dataset(X,y,file_names,ratio_train,val_test_split):
    

#     dataset = {"X":np.array(X),"y":np.array(y),"file_names":np.array(file_names)}
    
#     train_dataset = {"X":[],"y":[],"file_names":[]}
#     validation_dataset = {"X":[],"y":[],"file_names":[]}
#     test_dataset = {"X":[],"y":[],"file_names":[]}
    
#     for label in np.unique(dataset["y"]):
        
#         label_file_names = np.unique(np.extract(dataset["y"]==label,dataset["file_names"]))
        
#         for file_name in label_file_names:
            
#             indices = np.argwhere(dataset["file_names"]==file_name).flatten()
            
            
            
#             train_indices, val_indices = train_test_split(indices,
#                                                           train_size=ratio_train,
#                                                           shuffle=True)
            
#             val_indices, test_indices = train_test_split(val_indices,
#                                                           train_size=val_test_split,
#                                                           shuffle=True)
            
#             for key,value in dataset.items():
                
#                 train_data = dataset[key][train_indices].tolist()
#                 validation_data = dataset[key][val_indices].tolist()
#                 test_data = dataset[key][test_indices].tolist()
                
#                 train_dataset[key].extend(train_data)
#                 validation_dataset[key].extend(validation_data)
#                 test_dataset[key].extend(test_data)
            
#     train_dataset = shuffle_train_data(train_dataset) 
#     validation_dataset = shuffle_train_data(validation_dataset) 
#     test_dataset = shuffle_train_data(test_dataset)

#     return train_dataset, validation_dataset, test_dataset