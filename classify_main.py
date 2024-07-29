
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

from gapseqml.fileIO import (import_gapseqml_data, split_datasets, 
                             import_json_datasets, import_gapseq_simulated, import_json_evaluation_dataset)
from gapseqml.visualise import visualise_dataset
from gapseqml.classify_trainer import Trainer

# device
if torch.cuda.is_available():
    print("Training on GPU")
    device = torch.device('cuda:0')
else:
    print("Training on CPU")
    device = torch.device('cpu')


ratio_train = 0.9
val_test_split = 0.9
BATCH_SIZE = 10
LEARNING_RATE = 0.001
EPOCHS = 10
AUGMENT = True
NUM_WORKERS = 10
MODEL_FOLDER = "gapseqml_classification"

ml_data = import_gapseq_simulated(r"data\train\simulated")

# visualise_dataset(ml_data, n_examples = 3, label = 0, 
#                   n_rows = 4, n_cols = 4)


if __name__ == '__main__':

    datasets = split_datasets(ml_data, ratio_train, val_test_split)
    train_dataset, validation_dataset, test_dataset = datasets
    
    num_classes = len(np.unique(ml_data["labels"]))

    MODEL_FOLDER = f"gapseqml_simulated"

    model = InceptionTime(1,num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    trainer = Trainer(
        model=model,
        num_classes = num_classes,
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

    # trainer.visualise_augmentations(n_examples=5,
    #                                 show_plots=True,
    #                                 save_plots = False)
    
    
    # trainer.tune_hyperparameters(num_trials=50,
    #                               num_traces = 1000,
    #                               num_epochs = 5)

    model_path, state_dict_best = trainer.train()
    
    # model_path = r"C:\Users\turnerp\PycharmProjects\gapseqml\models\gapseqml_simulated_240716_1235\inceptiontime_model_240716_1235"
    # # model_path = r"\Users\turnerp\PycharmProjects\gapseqml\models\gapseqml_simulated_240715_1352\inceptiontime_model_240715_1352"
    
    # # model_path = r"C:\Users\turnerp\PycharmProjects\gapseqml\models\gapseqml_simulated_240715_0952\inceptiontime_model_240715_0952"


    
    # # json_dir = r"C:\Users\turnerp\PycharmProjects\gapseqml\data\evaluate"
    # # evaluation_dataset = import_json_evaluation_dataset(json_dir, "donor")
    
    # # with open('evaluation_dataset.pickle', 'wb') as handle:
    # #     pickle.dump(evaluation_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # with open('evaluation_dataset.pickle', 'rb') as handle:
    #     evaluation_dataset = pickle.load(handle)
    
    # predictions = trainer.evaluate_json_dataset(evaluation_dataset, model_path)
    # predictions.to_csv("predictions.csv",sep=",")
    
