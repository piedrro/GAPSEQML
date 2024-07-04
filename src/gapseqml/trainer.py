
import numpy as np
import torch
import tqdm
import torch.nn.functional as F
from skimage import exposure
from datetime import datetime
import os
import pathlib
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import copy
import warnings
from gapseqml.dataloader import load_dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import optuna
import traceback
import pathlib



class Trainer:

    def __init__(self,
                 model: torch.nn.Module = None,
                 pretrained_model=None,
                 device: torch.device = None,
                 criterion: torch.nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 train_dataset: dict = {},
                 validation_dataset: dict = {},
                 test_dataset: dict = {},
                 batch_size: int = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 tensorboard=bool,
                 epochs: int = 100,
                 kfolds: int = 0,
                 fold: int = 0,
                 model_folder = '',
                 model_path = None,
                 save_dir = '',
                 timestamp = datetime.now().strftime("%y%m%d_%H%M"),
                 ):

        self.model = model
        self.pretrained_model = pretrained_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
        self.model_folder = model_folder
        self.save_dir = save_dir
        self.model_path = model_path
        self.epoch = 0
        self.tensorboard = tensorboard
        self.training_loss = []
        self.training_accuracy = []
        self.validation_loss = []
        self.validation_accuracy = []
        self.learning_rate = []
        self.kfolds = kfolds
        self.fold = fold
        self.timestamp = timestamp
        self.best_epoch = 0
        self.best_model_weights = None
        
        if os.path.exists(save_dir):
            self.model_dir = os.path.join(save_dir,"models", model_folder + "_" + self.timestamp)
        else:
            self.model_dir = os.path.join("models", model_folder + "_" + self.timestamp)
            
        self.model_dir = os.path.abspath(self.model_dir)
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if pretrained_model:
            if os.path.isfile(pretrained_model):
                model_weights = torch.load(os.path.abspath(pretrained_model))['model_state_dict']
                model.load_state_dict(model_weights)
                
        if tensorboard:
            self.writer = SummaryWriter(log_dir= "runs/" + self.model_folder + "_" + timestamp)
            
        self.model_path = os.path.join(self.model_dir, f"inceptiontime_model_{self.timestamp}")
        
        self.initialise_dataloaders()

        
    def initialise_dataloaders(self):
        
        if hasattr(self, "train_dataset"):
            
            self.initialise_dataloader(self.train_dataset, "trainloader", 
                                       augment=True, batch_size=self.batch_size)
        

        if hasattr(self, "validation_dataset"):

            self.initialise_dataloader(self.validation_dataset, "valoader", 
                                       augment=False, batch_size=self.batch_size)
            
        if hasattr(self, "test_dataset"):

            self.initialise_dataloader(self.test_dataset, "testloader", 
                                       augment=False, batch_size=self.batch_size)
        
        
    def initialise_dataloader(self, dataset, name = "testloader", 
                              augment = False, batch_size = 10, shuffle=True):
        
        try:
        
            dataset = load_dataset(data = dataset["data"],
                               labels = dataset["labels"],
                               augment=augment)
    
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=False)
            
            setattr(self, name, dataloader)
            
            print(f"initialised {name} dataloader")
            
            n_images = len(dataloader)*batch_size
            setattr(self, f"num_{name.replace('loader','')}_images", n_images)
            
            return dataloader
            
        except:
            print(traceback.format_exc())
            
            return None
        
    def visualise_augmentations(self,  n_examples = 1, save_plots=True, show_plots=False):

        model_dir = pathlib.Path(self.model_dir)

        for example_int in range(n_examples):

            from random import randint
            random_index = randint(0, len(self.train_dataset["data"])-1)

            dataset = load_dataset(
                data=[self.train_dataset["data"][random_index]]*25,
                labels=[self.train_dataset["labels"][random_index]]*25,
                augment=True)
            
            dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

            centre_trace = self.train_dataset["data"][random_index]
            
            augmented_traces = []
            
            for data, _ in dataloader:

                data = data[0][0].numpy()
                augmented_traces.append(data)
                
            fig, ax = plt.subplots(3, 3, figsize=(18, 10))
            for i in range(3):
                for j in range(3):
                    index = i * 3 + j
                    if i == 1 and j == 1:
                        ax[i, j].plot(centre_trace, color='red')
                    else:
                        ax[i, j].plot(augmented_traces[index], color='blue')
                    ax[i, j].axis('off')
            
            fig.suptitle('Example Augmentations', fontsize=16)
            fig.tight_layout()

            if save_plots:
                plot_save_path = pathlib.Path('').joinpath(*model_dir.parts, "example_augmentations", f"example_augmentation{example_int}.tif")
                if not os.path.exists(os.path.dirname(plot_save_path)):
                    os.makedirs(os.path.dirname(plot_save_path))
                plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)

            if show_plots:
                plt.show()
            plt.close()
        
    
    def correct_predictions(self, label, pred_label):
    
        if len(label.shape) > 1:
            correct = (label.data.argmax(dim=1) == pred_label.data.argmax(dim=1)).float().sum().cpu()
        else:
            correct = (label.data == pred_label.data).float().sum().cpu()

        accuracy = correct / label.shape[0]

        return accuracy.numpy()


    def optuna_objective(self, trial):

        batch_size = trial.suggest_int("batch_size", 10, 100, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1)

        tune_trainloader = DataLoader(dataset=self.tune_train_dataset, batch_size=batch_size, shuffle=False)
        tune_valoader = DataLoader(dataset=self.tune_val_dataset, batch_size=batch_size, shuffle=False)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

        model = copy.deepcopy(self.model)
        model.to(self.device)

        running_loss = 0.0

        for data, labels in tune_trainloader:
            data, labels = data.to(self.device), labels.to(self.device)
            if not torch.isnan(data).any():
                self.optimizer.zero_grad()
                pred_label = model(data)
                loss = self.criterion(pred_label, labels)
                loss.backward()
                self.optimizer.step()

        for data, labels in tune_valoader:
            data, labels = data.to(self.device), labels.to(self.device)
            if not torch.isnan(data).any():
                pred_label = model(data)
                loss = self.criterion(pred_label, labels)
                running_loss += loss.item()

        return running_loss/len(tune_valoader)


    def load_tune_dataset(self, num_traces=100, num_epochs = 10):

        tune_images = []
        tune_labels = []
        
        tune_dataset = {"data": self.train_dataset["data"][:num_traces].copy(),
                        "labels": self.train_dataset["labels"][:num_traces].copy()}
        
        self.initialise_dataloader(tune_dataset, "tuneloader", 
                                   augment=True, batch_size=self.batch_size)

        for i in range(num_epochs):
            for images, labels in self.tuneloader:
                for image in images:
                    image = image.numpy()
                    tune_images.extend(image)
                for label in labels:
                    label = int(label.argmax(dim=0).numpy())
                    tune_labels.append(label)

        print(f"Loaded {len(tune_images)} traces for hyperparameter tuning.")
        
        tune_train_data = {"data": tune_images,
                              "labels": tune_labels}
        
        tune_val_data = {"data": self.validation_dataset["data"][:num_traces].copy(),
                          "labels": self.validation_dataset["labels"][:num_traces].copy()}
        
        self.tune_train_dataset = load_dataset(data = tune_train_data["data"],
                                                labels = tune_train_data["labels"],
                                                augment=False)
        
        self.tune_val_dataset = load_dataset(data = tune_val_data["data"],
                                                labels = tune_val_data["labels"],
                                                augment=False)
        
    def tune_hyperparameters(self, num_trials=5, num_traces = 500, num_epochs = 4):

        self.load_tune_dataset(num_traces=num_traces, num_epochs=num_epochs)

        self.num_tune_traces = num_traces
        self.num_tune_epochs = num_epochs

        study = optuna.create_study(direction='minimize')
        study.optimize(self.optuna_objective, n_trials=num_trials)

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        self.batch_size = int(trial.params["batch_size"])
        self.learning_rate = float(trial.params["learning_rate"])
        self.hyperparameter_study = study
        
        model_dir = pathlib.Path(self.model_dir)

        optimisation_history_path = pathlib.Path('').joinpath(*model_dir.parts, "Optuna","optuna_optimisation_history_plot.png")
        slice_plot_path = pathlib.Path('').joinpath(*model_dir.parts, "Optuna","optuna_slice_plot.png")
        parallel_coordinate_plot_path = pathlib.Path('').joinpath(*model_dir.parts, "Optuna","optuna_parallel_coordinate_plot.png")
        contour_plot_path = pathlib.Path('').joinpath(*model_dir.parts, "Optuna","optuna_contour_plot.png")
        param_importances_plot_path = pathlib.Path('').joinpath(*model_dir.parts, "Optuna","optuna_param_importances_plot.png")

        if not os.path.exists(os.path.dirname(optimisation_history_path)):
            os.makedirs(os.path.dirname(optimisation_history_path))

        optuna.visualization.plot_optimization_history(study).write_image(optimisation_history_path)
        optuna.visualization.plot_slice(study).write_image(slice_plot_path)
        optuna.visualization.plot_parallel_coordinate(study).write_image(parallel_coordinate_plot_path)
        optuna.visualization.plot_contour(study).write_image(contour_plot_path)
        optuna.visualization.plot_param_importances(study).write_image(param_importances_plot_path)

        from PIL import Image
        img = np.asarray(Image.open(slice_plot_path))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        return study


    def train(self):

        progressbar = tqdm.tqdm(range(self.epochs), 'Progress', total=self.epochs, position=0, leave=True)

        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self.train_step()

            """Validation block"""
            if self.valoader is not None:
                self.val_step()

            """update tensorboard"""
            if self.writer:
                self.writer.add_scalar("Loss/train", self.training_loss[-1], self.epoch)
                self.writer.add_scalar("Loss/validation", self.validation_loss[-1], self.epoch)
                self.writer.add_scalar("Accuracy/train", self.training_accuracy[-1], self.epoch)
                self.writer.add_scalar("Accuracy/validation", self.validation_accuracy[-1], self.epoch)

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()  # learning rate scheduler step

            if self.validation_loss[-1] == np.min(self.validation_loss):
                
                self.best_epoch = self.epoch
                self.best_model_weights = self.model.state_dict()
                
            torch.save({'best_epoch': self.best_epoch,
                        'num_epochs': self.epochs,
                        'model_state_dict': self.best_model_weights,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'lr_scheduler': self.lr_scheduler,
                        'training_loss': self.training_loss,
                        'validation_loss': self.validation_loss,
                        'training_accuracy': self.training_accuracy,
                        'validation_accuracy': self.validation_accuracy,
                        'num_validation_images': self.num_validation_images}, self.model_path)

            progressbar.set_description(
                f'(Training Loss {self.training_loss[-1]:.5f}, Validation Loss {self.validation_loss[-1]:.5f})')  # update progressbar

        return self.model_path, self.best_model_weights

    def train_step(self):

        train_losses = []  # accumulate the losses here
        train_accuracies = []

        batch_iter = tqdm.tqdm(enumerate(self.trainloader), 'Training', total=len(self.trainloader), position=1, leave=True)

        for i, (images, labels) in batch_iter:
            images, labels = images.to(self.device), labels.to(self.device)  # send to device (GPU or CPU)

            self.optimizer.zero_grad()  # zerograd the parameters
            pred_label = self.model(images)  # one forward pass

            loss = self.criterion(pred_label, labels)
            train_losses.append(loss.item())

            accuracy = self.correct_predictions(pred_label, labels)
            train_accuracies.append(accuracy)

            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            current_lr = self.optimizer.param_groups[0]['lr']

            batch_iter.set_description(
                f'Training: (loss {np.mean(train_losses):.5f}, Acc {np.mean(train_accuracies):.2f} LR {current_lr})')  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.training_accuracy.append(np.mean(train_accuracies))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def val_step(self):

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        valid_accuracies = []

        batch_iter = tqdm.tqdm(enumerate(self.valoader), 'Validation', total=len(self.valoader), position=1, leave=True)

        for i, (images, labels) in batch_iter:
            images, labels = images.to(self.device), labels.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                pred_label = self.model(images)

                loss = self.criterion(pred_label, labels)
                valid_losses.append(loss.item())

                accuracy = self.correct_predictions(pred_label, labels)
                valid_accuracies.append(accuracy)

                current_lr = self.optimizer.param_groups[0]['lr']

                batch_iter.set_description(
                    f'Validation: (loss {np.mean(valid_losses):.5f}, Acc {np.mean(valid_accuracies):.2f} LR {current_lr})')  # update progressbar

        self.validation_loss.append(np.mean(valid_losses))
        self.validation_accuracy.append(np.mean(valid_accuracies))

        batch_iter.close()
        
    
    def evaluate(self, testloader, model_path = None):
        
        if os.path.isfile(model_path) == True:
            
            self.model_path = model_path
            
            model_data = torch.load(model_path)
            model_weights = model_data['model_state_dict']
            self.model.load_state_dict(model_weights)
            
        else:
            model_data = {}
            
        self.model.eval()  # evaluation mode

        saliency_maps = []
        true_labels = []
        pred_labels = []
        pred_losses = []
        test_data = []
        pred_confidences = []
         
        batch_iter = tqdm.tqdm(enumerate(testloader), 'Evaluating', total=len(testloader), position=1, leave=True)
        
        for i, (images, labels) in batch_iter:
            
            traces, labels = images.to(self.device), labels.to(self.device)  # send to device (GPU or CPU)
        
            with torch.no_grad():
                
                pred_label = self.model(traces)
                loss = self.criterion(pred_label, labels)

                pred_confidences.extend(torch.nn.functional.softmax(pred_label, dim=1).tolist())
                pred_labels.extend(pred_label.data.cpu().argmax(dim=1).numpy().tolist())
                true_labels.extend(labels.data.cpu().argmax(dim=1).numpy().tolist())
                pred_losses.append(loss.item())
                test_data.extend(traces.data.cpu().numpy().tolist())
                
        batch_iter.close()  
        
        test_accuracy = np.sum(np.array(pred_labels) == np.array(true_labels))/len(pred_labels)
        
        pred_confidences = np.array(pred_confidences).max(axis=-1).tolist()
        
        cm = confusion_matrix(true_labels, pred_labels, normalize='pred')
        
        model_data["test_results"] = {}
        
        model_data["test_results"]["confusion_matrix"] = cm
        model_data["test_results"]["true_labels"] = true_labels
        model_data["test_results"]["pred_labels"] = pred_labels
        model_data["test_results"]["test_accuracy"] = test_accuracy
        model_data["test_results"]["test_data"] = test_data
        model_data["test_results"]["pred_confidences"] = pred_confidences
        
        torch.save(model_data, self.model_path)
        
        return model_data

        