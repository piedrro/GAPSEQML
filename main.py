
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
from gapseqml.dataloader import load_dataset


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

# datasets = split_datasets(ml_data, ratio_train, val_test_split)
# train_dataset, validation_dataset, test_dataset = datasets

# if __name__ == '__main__':

#     training_dataset = load_dataset(data = train_dataset["data"],
#                                     labels = train_dataset["labels"],
#                                     augment = True)

#     validation_dataset = load_dataset(data = validation_dataset["data"],
#                                     labels = validation_dataset["labels"],
#                                       augment=False)

#     test_dataset = load_dataset(data = test_dataset["data"],
#                                 labels = test_dataset["labels"],
#                                 augment=False)

#     trainloader = data.DataLoader(dataset=training_dataset,
#                                   batch_size=BATCH_SIZE,
#                                   shuffle=True)

#     valoader = data.DataLoader(dataset=validation_dataset,
#                                 batch_size=BATCH_SIZE,
#                                 shuffle=False)

#     testloader = data.DataLoader(dataset=test_dataset,
#                                   batch_size=BATCH_SIZE,
#                                   shuffle=False)

#     model = InceptionTime(1,len(np.unique(train_dataset["labels"]))).to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
#     timestamp = datetime.now().strftime("%y%m%d_%H%M")

#     trainer = Trainer(model=model,
#               device=device,
#               optimizer=optimizer,
#               criterion=criterion,
#               trainloader=trainloader,
#               valoader=valoader,
#               lr_scheduler=scheduler,
#               tensorboard=True,
#               epochs=EPOCHS,
#               batch_size = BATCH_SIZE,
#               model_folder=MODEL_FOLDER)

#     model_path, state_dict_best = trainer.train()

#     model_data = trainer.evaluate(testloader, model_path)
    
    
    
    