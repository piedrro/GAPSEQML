
import numpy as np
from skimage import exposure
import itertools
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import io
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import os
import torch
from gapseqml.dataloader import load_dataset
from torch.utils.data import DataLoader
import traceback



def visualise_dataset(dataset, n_examples = 1, label = 1, n_rows = 3, n_cols = 3):

    data = np.array(dataset["data"])
    labels = np.array(dataset["labels"])

    if label in labels:
        label_indices = np.argwhere(labels==label).flatten()

        vis_data = data[label_indices]
        vis_labels = labels[label_indices]
    else:
        vis_data = data
        vis_labels = labels
        
    dataset = load_dataset(data=vis_data.tolist(),
                           labels=vis_labels.tolist(),
                           augment=False)
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=n_cols*n_rows,
                            shuffle=True)
    
    for _ in range(n_examples):
        
        plot_data = []
        
        for data, labels in dataloader:
            
            dat = np.squeeze(data).tolist()
            plot_data.extend(dat)
            
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(18, 10))
        for i in range(n_rows):
            for j in range(n_cols):
                index = i * n_rows + j
                ax[i, j].plot(plot_data[index], color='blue')
                ax[i, j].axis('off')
            
        fig.tight_layout()
        plt.show()
            
    return vis_labels, vis_data



def plot_confusion_matrix(true_labels, pred_labels, classes, 
                          num_samples=1, title='Confusion matrix', 
                          cmap=plt.cm.Blues, colourbar = True, save_path=None):
    
    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = confusion_matrix(true_labels, pred_labels, normalize="true")

    accuracy = len(np.where(np.array(true_labels) == np.array(pred_labels))[0]) / len(true_labels)
    balanced_accuracy = balanced_accuracy_score(true_labels, pred_labels)

    plt.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    plt.title(title + "\n" + f"N: {len(true_labels)}\nAccuracy: {accuracy:.2f}\nBalanced Accuracy: {balanced_accuracy:.2f}")

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=90, ha='center', rotation_mode='anchor')
    plt.tick_params(axis='y', which='major', pad=10)
    
    if colourbar:
        plt.colorbar()

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        accuracy = cm_norm[i,j]    
        num_labels = cm[i,j]

        plt.text(j, i, f"{accuracy:.2f}" + " (" + str(num_labels) + ")", 
                 horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()

    if save_path:
         plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
         plt.show()
         image = Image.open(save_path)
         ar = np.asarray(image)
    else:
        with io.BytesIO() as buffer:
            plt.savefig(buffer, format="png", bbox_inches='tight', pad_inches=0, dpi=300)
            buffer.seek(0)
            image = Image.open(buffer)
            ar = np.asarray(image)
            plt.show()

    plt.close()

    return ar

def plot_evaluation_visualisations(model_data, save_path, 
                                   class_labels = ["Comp","Non-Comp"]):
    
    pred_labels = model_data["test_results"]["pred_labels"]
    true_labels = model_data["test_results"]["true_labels"]
    
    class_labels = ["Comp","Non-Comp"]
    
    cm_path = save_path + "_confusion_matrix.tif"
    loss_graph_path = save_path + "_loss_graph.tif"
    accuracy_graph_path = save_path + "_accuracy_graph.tif"

    plot_confusion_matrix(true_labels, pred_labels, class_labels, 
                          save_path = cm_path)
    
    plot_confidence_plots(model_data, save_path=save_path)
    
    plot_prediction_histograms(model_data, save_path=save_path)
    
    train_loss = model_data["training_loss"]
    validation_loss = model_data["validation_loss"]
    train_accuracy = model_data["training_accuracy"]
    validation_accuracy = model_data["validation_accuracy"]
    
    plt.plot(train_loss, label="training loss")
    plt.plot(validation_loss, label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Binary Cross Entropy Loss")
    plt.legend(loc="upper right")
    plt.savefig(loss_graph_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    plt.plot(train_accuracy, label="training accuracy")
    plt.plot(validation_accuracy, label="validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="lower right")
    plt.savefig(accuracy_graph_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    

def get_confidence_indices(pred_confidences, pred_labels, true_labels):
    
    pred_confidences = np.array(pred_confidences)
    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)
    
    # Identify correct and incorrect predictions
    correct_indices = np.where(pred_labels == true_labels)[0]
    incorrect_indices = np.where(pred_labels != true_labels)[0]

    # Get confidences for correct and incorrect predictions
    correct_confidences = pred_confidences[correct_indices]
    incorrect_confidences = pred_confidences[incorrect_indices]

    # Sort correct predictions by confidence
    sorted_correct_indices = correct_indices[np.argsort(correct_confidences)]
    most_confident_correct = sorted_correct_indices[::-1]  # Most confident first
    least_confident_correct = sorted_correct_indices  # Least confident first

    # Sort incorrect predictions by confidence
    sorted_incorrect_indices = incorrect_indices[np.argsort(incorrect_confidences)]
    most_confident_incorrect = sorted_incorrect_indices[::-1]  # Most confident first
    least_confident_incorrect = sorted_incorrect_indices  # Least confident first
    
    indices = {"Most Confident True Predictions":most_confident_correct,
               "Least Confident True Predictions":least_confident_correct,
               "Most Confident False Predictions":most_confident_incorrect,
               "Least Confident False Predictions":least_confident_incorrect}

    return indices


def plot_confidence_plots(model_data, save_path=None):
    
    plot_images = {}

    test_results = model_data["test_results"]
    
    pred_confidences = test_results["pred_confidences"]
    pred_labels = np.array(test_results["pred_labels"])
    true_labels = np.array(test_results["true_labels"])
    test_data = np.array(test_results["test_data"])
    
    n_rows = 3
    n_cols = 3
    
    n_plots = n_rows*n_cols
    
    index_dict = get_confidence_indices(pred_confidences, pred_labels, true_labels)
    
    for index_name, indices in index_dict.items():
        
        try:

            save_name = "_".join(index_name.lower().split(" ")) + ".tif"
            
            if save_path is not None:
                plot_path = save_path + save_name
            else:
                plot_path = None
                
            indices = indices[:n_plots]
            
            fig, ax = plt.subplots(n_rows, n_cols, figsize=(18, 10))
            fig.suptitle(index_name, fontsize=24)
            
            for i in range(n_rows):
                for j in range(n_cols):
                    
                    plot_index = i * n_rows + j
                    data_index = indices[plot_index]
                    
                    plot_data = test_data[data_index][0]
                    plot_conf = pred_confidences[data_index]
                    plot_label = true_labels[data_index]
                    
                    ax[i, j].plot(plot_data, color='blue')
                    ax[i, j].axis('off')
                    
                    plot_text = f"Label: {plot_label}\nConfidence: {plot_conf:.4f}"
                    
                    ax[i, j].text(0.05, 0.95, plot_text, 
                      horizontalalignment='left', 
                      verticalalignment='top', 
                      transform=ax[i, j].transAxes, 
                      fontsize=15,
                      color='black',
                      fontweight='bold',
                      bbox=dict(facecolor='grey', alpha=0.7, edgecolor='none')
                      )
                    
            plt.tight_layout()
    
            if plot_path:
                  plt.savefig(plot_path, bbox_inches='tight', pad_inches=0, dpi=300)
                  plt.show()
                  image = Image.open(plot_path)
                  ar = np.asarray(image)
            else:
                with io.BytesIO() as buffer:
                    plt.savefig(buffer, format="png", bbox_inches='tight', pad_inches=0, dpi=300)
                    buffer.seek(0)
                    image = Image.open(buffer)
                    ar = np.asarray(image)
                    plt.show()
                    
                    plot_images[save_name] = ar
                    
        except:
            print(traceback.format_exc())
                
    return index_dict
                
        
def plot_prediction_histograms(model_data, save_path=None):
    
    try:

        test_results = model_data["test_results"]
        
        pred_confidences = np.array(test_results["pred_confidences"])
        pred_labels = np.array(test_results["pred_labels"])
        true_labels = np.array(test_results["true_labels"])
        
        for label in np.unique(true_labels):
            
            label_indices = np.where((true_labels == label))[0]
            
            confidences = pred_confidences[label_indices]
            plt.hist(confidences, bins=20, alpha=0.5, label=f'Label {label}')
            
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Prediction Confidence Histogram')
        if save_path is not None:
            plot_path = save_path + "prediction_histogram.tif"
            plt.savefig(plot_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show()
            
        for label in np.unique(true_labels):
            correct_label_indices = np.where((pred_labels == true_labels) &
                                              (true_labels == label))[0]
            
            confidences = pred_confidences[correct_label_indices]
            plt.hist(confidences, bins=20, alpha=0.5, label=f'Label {label}')
            
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Confidence Histograms for Correctly Classified Labels')
        if save_path is not None:
            plot_path = save_path + "correct_prediction_histogram.tif"
            plt.savefig(plot_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show()
        
        for label in np.unique(true_labels):
            incorrect_label_indices = np.where((pred_labels != true_labels) &
                                              (true_labels == label))[0]
            
            confidences = pred_confidences[incorrect_label_indices]
            plt.hist(confidences, bins=20, alpha=0.5, label=f'Label {label}')
            
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Confidence Histograms for Incorrectly Classified Labels')
        if save_path is not None:
            plot_path = save_path + "incorrect_prediction_histogram.tif"
            plt.savefig(plot_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show()
        
    except:
        print(traceback.format_exc())
        pass



# model_path = r"C:\Users\turnerp\PycharmProjects\gapseqml\models\TEST_240705_0932\inceptiontime_model_240705_0932"
# model_data = torch.load(model_path)


# test_results = model_data["test_results"]
# pred_labels = np.array(test_results["pred_labels"])
# true_labels = np.array(test_results["true_labels"])

# plot_confusion_matrix(true_labels, pred_labels, ["comp","non-comp"])

# plot_evaluation_visualisations(model_data, model_path)
# index_dict = plot_confidence_plots(model_data)
        

# correct_indices = np.argwhere(pred_labels==true_labels)
# incorred_indices = np.argwhere(pred_labels!=true_labels)

# highest_conf_indices = np.argsort(confidences)[-n_plots:][::-1]
# lowest_conf_indices = np.argsort(confidences)[:n_plots]


# plot_data = []
# plot_conf = []
# plot_labels = []

# for index in highest_conf_indices:
    
#     data = test_results["test_data"][index][0]
#     label = test_results["true_labels"][index]
#     conf = confidences[index]
    
#     plot_data.append(data)
#     plot_labels.append(label)
#     plot_conf.append(conf)









    
    
    



