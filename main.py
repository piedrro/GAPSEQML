
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
from datetime import datetime


from dataloader import load_dataset
from file_io import read_gapseq_data
from trainer import Trainer
import random

import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim

from tsai.all import InceptionTime


# device
if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')
      

ratio_train = 0.8
val_test_split = 0.5
BATCH_SIZE = 10
LEARNING_RATE = 0.001
EPOCHS = 5
AUGMENT = True
MODEL_FOLDER = "TEST"


def jittering(x, sigma=0.03):
    
    x = x + np.random.normal(loc=0., scale=sigma, size=x.shape)
    
    return x

def scaling(x, sigma=0.1):
    
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0]))
    
    x = np.multiply(x, factor)
    
    return x

def rolling(x):

    start = np.random.randint(0, len(x))
    
    x = np.roll(x, start,0)
    
    return x
    
def slicing(x, slice_range = [50,100]):
    
    slice_percent = np.random.randint(slice_range[0],slice_range[1])
    
    slice_length = int((len(x)/100)*slice_percent)
    
    x = random.sample(x, slice_length)
    
    x = np.interp(np.linspace(0, len(x) - 1, num=1200), np.arange(len(x)), x)
    
    return x
    
def flipping(x):
    
    flip = bool(random.getrandbits(1))
    
    if flip:
    
        x = np.flip(x)
    
    return x

    



if __name__ == '__main__':
        
    # directory_path = r"/run/user/26623/gvfs/smb-share:server=physics.ox.ac.uk,share=dfs/DAQ/CondensedMatterGroups/AKGroup/Jagadish/Traces for ML"
    
    # complimentary_files_path = os.path.join(directory_path, "Complementary Traces")
    # noncomplimentary_files_path = os.path.join(directory_path, "Non_Complementary Traces")
    
    # complimentary_files = glob(complimentary_files_path + "*/*_gapseqML.txt")
    # noncomplimentary_files = glob(noncomplimentary_files_path + "*/*_gapseqML.txt")
    
    # X, y, file_names = read_gapseq_data(complimentary_files, X, y, file_names, 0)
    # X, y, file_names = read_gapseq_data(noncomplimentary_files, X, y, file_names, 1)
    
    
    # with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([X, y, file_names], f)
    
    
    with open('objs.pkl','rb') as f:  # Python 3: open(..., 'rb')
        X, y, file_names = pickle.load(f)
        
        
    X, y, file_names = shuffle(X, y, file_names) 
    
    
    # X = [X[0]]*len(X)
    
    # slice_percent = 80
    
    # slice_length = int((len(x)/100)*slice_percent)
    
    # x = random.sample(x, slice_length)
    
    # x = np.interp(np.linspace(0, len(x) - 1, num=1200), np.arange(len(x)), x)
    
    
    
    # x = np.expand_dims(np.expand_dims(np.array(X[0]),0),0)
    
    # x = window_slice(x)
    
    
    # for i in range(10):
        
    #     slicex = flipping(x)
        
    #     plt.plot(slicex)
    #     plt.show()
    
    



    
    # from scipy.interpolate import CubicSpline
    
    # sigma=0.2
    # knot=4
    
    
    # random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, x.shape[0]))
    
    
    
    
    # for i in range(10):
        
    #     axis = np.random.randint(0, len(x))
        
    #     x = np.roll(x, axis,0)
        
    #     print(axis)
        
    #     # x = scaling(x)
        
    #     plt.plot(x)
    #     plt.show()
        
        
    
    
    
    
    
    
    
    
    
    X_train, X_val, y_train, y_val = train_test_split(np.array(X),
                                                        np.array(y),
                                                        train_size=0.8,
                                                        shuffle=True)
    
    training_dataset = load_dataset(data = X_train,
                                    labels = y_train,
                                    augment = True)
    validation_dataset = load_dataset(data = X_val,
                                      labels = y_val)
    
    trainloader = data.DataLoader(dataset=training_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers = 20)
    
    valoader = data.DataLoader(dataset=validation_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False, num_workers = 20)
    
    # traces,label = next(iter(trainloader))
    
    
    
    
    # for X in traces:
        
    #     plt.plot(X[0])
    #     plt.show()
    
    
    model = InceptionTime(1,len(np.unique(y))).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    
    trainer = Trainer(model=model,
              device=device,
              optimizer=optimizer,
              criterion=criterion,
              trainloader=trainloader,
              valoader=valoader,
              lr_scheduler=scheduler,
              tensorboard=True,
              epochs=EPOCHS,
              batch_size = BATCH_SIZE,
              model_folder=MODEL_FOLDER)
    
    model_path = trainer.train()
    
    model_data = torch.load(model_path)


    
    
    
    
    
    
    
    
    
    