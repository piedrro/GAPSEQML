
from glob2 import glob
import pandas as pd
from datetime import datetime 
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import os
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from torch.utils import data
from tsai.all import InceptionTime

from dataloader import load_dataset
from file_io import read_gapseq_data
from trainer import Trainer

# device
if torch.cuda.is_available():
    print("Training on GPU")
    device = torch.device('cuda:0')
else:
    print("Training on CPU")
    device = torch.device('cpu')
      

ratio_train = 0.7
val_test_split = 0.5
BATCH_SIZE = 10
LEARNING_RATE = 0.0001
EPOCHS = 2
AUGMENT = True
NUM_WORKERS = 10
MODEL_FOLDER = "TEST"



directory_path = r"/home/turnerp/.cache/gvfs/smb-share:server=physics.ox.ac.uk,share=dfs/DAQ/CondensedMatterGroups/AKGroup/Jagadish/Traces for ML_GAP-sequencing"

complimentary_files_path = os.path.join(directory_path, "Complementary Traces")
noncomplimentary_files_path = os.path.join(directory_path, "Non_Complementary Traces")

complimentary_files = glob(complimentary_files_path + "*/*_gapseqML.txt")
noncomplimentary_files = glob(noncomplimentary_files_path + "*/*_gapseqML.txt")


X, y, file_names = read_gapseq_data(complimentary_files, label=0, trace_limit=1200)
X, y, file_names = read_gapseq_data(noncomplimentary_files, X, y, file_names, label=1, trace_limit=1200)


if __name__ == '__main__':
    
    

    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      train_size=ratio_train,
                                                      random_state=42,
                                                      shuffle=True)

    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val,
                                                    train_size=val_test_split,
                                                    random_state=42,
                                                    shuffle=True)
    
    training_dataset = load_dataset(data = X_train,
                                    labels = y_train,
                                    augment = True)
    
    validation_dataset = load_dataset(data = X_val,
                                    labels = y_val,
                                      augment=False)
    
    test_dataset = load_dataset(data = X_test,
                                labels = y_test,
                                augment=False)
    
    trainloader = data.DataLoader(dataset=training_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    
    valoader = data.DataLoader(dataset=validation_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False)
    
    testloader = data.DataLoader(dataset=test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False)
    
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
    
    model_path, state_dict_best = trainer.train()
    
    model_data = trainer.evaluate(testloader, model_path)
    
