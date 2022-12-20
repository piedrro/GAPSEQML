
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
BATCH_SIZE = 20
LEARNING_RATE = 0.0001
EPOCHS = 100
AUGMENT = True
MODEL_FOLDER = "TEST"


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
    
    
    X_train, X_val, y_train, y_val = train_test_split(np.array(X),
                                                        np.array(y),
                                                        train_size=0.8,
                                                        shuffle=True)
    
    training_dataset = load_dataset(data = X_train,
                                    labels = y_train,
                                    augment = True)
    
    validation_dataset = load_dataset(data = X_val,
                                      labels = y_val,
                                      augment=False)
    
    trainloader = data.DataLoader(dataset=training_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers = 20)
    
    valoader = data.DataLoader(dataset=validation_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False, num_workers = 20)
    
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


    
    
    
    
    
    
    
    
    
    