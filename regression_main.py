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

from gapseqml.dataloader import load_dataset

from gapseqml.fileIO import (import_gapseqml_data, split_datasets, 
                             import_json_datasets, import_gapseq_simulated, import_json_evaluation_dataset)

from gapseqml.visualise import visualise_dataset
from gapseqml.regression_trainer import Trainer


# device
if torch.cuda.is_available():
    print("Training on GPU")
    device = torch.device('cuda:0')
else:
    print("Training on CPU")
    device = torch.device('cpu')


ratio_train = 0.9
val_test_split = 0.9
BATCH_SIZE = 12
LEARNING_RATE = 0.0001
EPOCHS = 5
AUGMENT = True
NUM_WORKERS = 10
MODEL_FOLDER = "gapseqml_regression"

ml_data = import_gapseq_simulated(r"data\train\simulated")

class regressionModel(nn.Module):
    def __init__(self, c_in, c_out):
        super(regressionModel, self).__init__()
        self.inception_time = InceptionTime(c_in, c_out)
        self.fc = nn.Linear(c_out, 1)  # Output layer for regression

    def forward(self, x):
        x = self.inception_time(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    
    datasets = split_datasets(ml_data, ratio_train, val_test_split)
    train_dataset, validation_dataset, test_dataset = datasets
    
    model = regressionModel(c_in=1, c_out=64).to(device)
    
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    trainer = Trainer(
        model=model,
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
    
    # model = trainer.tune_hyperparameters()
    
    model_path, state_dict_best = trainer.train()
    
    # model_path = r"C:\Users\turnerp\PycharmProjects\gapseqml\models\gapseqml_regression_240725_1634\inceptiontime_model_240725_1634"
    
    # json_dir = r"C:\Users\turnerp\PycharmProjects\gapseqml\data\evaluate"
    # evaluation_dataset = import_json_evaluation_dataset(json_dir, "acceptor")
    
    # with open('evaluation_dataset.pickle', 'wb') as handle:
    #     pickle.dump(evaluation_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # with open('evaluation_dataset.pickle', 'rb') as handle:
    #     evaluation_dataset = pickle.load(handle)
    
    # predictions = trainer.evaluate_json_dataset(evaluation_dataset, model_path)
    
    