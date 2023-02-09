
from glob2 import glob
import pandas as pd
from datetime import datetime 
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from torch.utils import data
from tsai.all import InceptionTime

from dataloader import load_dataset
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



data_path = r"\\PHYSICS\dfs\DAQ\CondensedMatterGroups\AKGroup\anna\DeeplearningFRET\AutoSim_AAdata\allData.npy"
labels_path = r"\\PHYSICS\dfs\DAQ\CondensedMatterGroups\AKGroup\anna\DeeplearningFRET\AutoSim_AAdata\classificationData.npy"

X = np.load(data_path,allow_pickle=True)
y = np.load(labels_path,allow_pickle=True)

y = np.array(y).flatten().tolist()
X = np.array_split(X, len(X))
X = [dat[0] for dat in X]


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
    
    model = InceptionTime(3,len(np.unique(y))).to(device)
    
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
    
    trainer.evaluate(testloader, model_path)
    
