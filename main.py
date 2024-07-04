
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
import pickle

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
LEARNING_RATE = 0.001
EPOCHS = 3
AUGMENT = True
NUM_WORKERS = 10
MODEL_FOLDER = "TEST"


ml_data = {"data":[],"labels":[],"n_nucleotide":[],"file_names":[]}

ml_data = import_gapseqml_data(r"data/3nt/comp",
    label = 0, n_nucleotide=3, trace_length = 800, ml_data=ml_data)
ml_data = import_gapseqml_data(r"data/3nt/noncomp",
    label = 1, n_nucleotide=3, trace_length = 800, ml_data=ml_data)
# ml_data = import_gapseqml_data(r"data/5nt/comp",
#     label = 0, n_nucleotide=5, trace_length = 800, ml_data=ml_data)
# ml_data = import_gapseqml_data(r"data/5nt/noncomp",
#     label = 1, n_nucleotide=5, trace_length = 800, ml_data=ml_data)

with open('ml_data.pickle', 'wb') as handle:
    pickle.dump(ml_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('ml_data.pickle', 'rb') as handle:
    ml_data = pickle.load(handle)


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
              learning_rate = LEARNING_RATE,
              batch_size = BATCH_SIZE,
              model_folder=MODEL_FOLDER)
    
    trainer.visualise_augmentations(n_examples=5,
                                    show_plots=True,
                                    save_plots = True)
    
    trainer.tune_hyperparameters(num_trials=5, 
                                  num_traces = 200, 
                                  num_epochs = 5)
    
    model_path, state_dict_best = trainer.train()
    
    model_data = trainer.evaluate(test_dataset, model_path)
    
    
    
    