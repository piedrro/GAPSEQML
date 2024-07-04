
from glob2 import glob
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from tsai.all import InceptionTime
import json
import traceback
import sklearn

from gapseqml.fileIO import import_gapseqml_data, split_datasets
from gapseqml.trainer import Trainer

# device
if torch.cuda.is_available():
    print("Training on GPU")
    device = torch.device('cuda:0')
else:
    print("Training on CPU")
    device = torch.device('cpu')


ratio_train = 0.8
val_test_split = 0.8
BATCH_SIZE = 10
LEARNING_RATE = 0.0001
EPOCHS = 2
AUGMENT = True
NUM_WORKERS = 10
MODEL_FOLDER = "TEST"


comp_folders = [r"data/3nt/comp"]
noncomp_folders = [r"data/3nt/noncomp",]

ml_data = import_gapseqml_data(comp_folders, label = 0, trace_length = 800)
ml_data = import_gapseqml_data(noncomp_folders, label = 1, trace_length = 800, ml_data=ml_data)

datasets = split_datasets(ml_data, ratio_train, val_test_split)
train_dataset, validation_dataset, test_dataset = datasets

if __name__ == '__main__':


    model = InceptionTime(1,len(np.unique(train_dataset["labels"]))).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    trainer = Trainer(model=model,
              device=device,
              optimizer=optimizer,
              criterion=criterion,
              train_dataset=train_dataset,
              validation_dataset=validation_dataset,
              test_dataset=test_dataset,
              lr_scheduler=scheduler,
              tensorboard=True,
              epochs=EPOCHS,
              batch_size = BATCH_SIZE,
              model_folder=MODEL_FOLDER)
    
    # trainer.tune_hyperparameters(num_trials=5, 
    #                              num_traces = 200, 
    #                              num_epochs = 5)
    
    trainer.visualise_augmentations(n_examples=5, 
                                    show_plots=True, 
                                    save_plots = True)

    # model_path, state_dict_best = trainer.train()

    # model_data = trainer.evaluate(testloader, model_path)
    
    
    
    