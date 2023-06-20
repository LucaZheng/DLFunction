# for classifcation task. 

# System libraries
from pathlib import Path
import os.path
import random
import time

# Utilities
import pandas as pd, numpy as np

# Sklearn
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import precision_recall_curve,average_precision_score,roc_auc_score
from sklearn.preprocessing import label_binarize

# Visualization Libraries
import matplotlib.cm as cm
import cv2
import seaborn as sns
import matplotlib.pyplot as plt


# essential for plotting. passing final model, testing set(dataframe) and it ouputs probabilities, predicted lable, and true lable(array)
def predict_test(model,test_images):
  # Predict the label of the test_images
  pred = model.predict(test_images)

  # Map the label
  labels = (train_images.class_indices)
  labels = dict((v,k) for k,v in labels.items())
  pred_labels = np.argmax(pred,axis=1)
  pred_labels = np.array([labels[k] for k in pred_labels])

  true = np.array(test_df.Label.to_list())

  return pred_labels,pred,true

# plot roc with auc curve for multi-classes classification
def plot_rocau_mul(pred_labels, pred, true, model_name, show_all=None, show_micro_average=True ,plot_pr=True):

    # Create dic for birds name mapping
    bird_dics = {'bermuda_petrel':0,
    'black_winged_petrel':1,
    'cooks_petrel':2,
    'feas_petrel':3,
    'galapagos_petrel':4,
    'goulds_petrel':5,
    'hawaiian_petrel':6,
    'masatierra_petrel':7,
    'soft_plumaged_petrel':8}

    # mapping labels for both prediction and true
    numeric_preds = [bird_dics[bird] for bird in pred_labels]
    numeric_true_labels = [bird_dics[bird] for bird in true]

    # create binary labels for plotting
    y_test_binarized = label_binarize(numeric_true_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    y_pred_binarized = label_binarize(numeric_preds, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    n_classes = y_test_binarized.shape[1]

    # PLOT ROC-AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Mapping birds name back to 0-8
    reverse_bird_dics = {v: k for k, v in bird_dics.items()}

    # Prepare AUC for ordering
    auc_scores = [(reverse_bird_dics[i], roc_auc[i]) for i in range(n_classes)]
    auc_scores.sort(key=lambda x: x[1], reverse=True)

    # Prepare classes for displaying
    if show_all is not None:
        show_classes = [bird_dics[bird] for bird in show_all]
    else:
        show_classes = list(range(n_classes))

    # Plot ROC-AUC
    plt.figure()
    for i in show_classes:
        plt.plot(fpr[i], tpr[i], label='{0} (area = {1:0.2f})'.format(*auc_scores[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name + ' Multi-Class ROC-AUC Cruves')
    plt.legend(loc="lower right")
    plt.show()

    # PLOT PR CURVE
    if plot_pr:
    # For each class
      precision = dict()
      recall = dict()
      average_precision = dict()
      for i in range(n_classes):
          precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i],
                                                            y_pred_binarized[:, i])
          average_precision[i] = average_precision_score(y_test_binarized[:, i], y_pred_binarized[:, i])

    # A "micro-average": quantifying score on all classes jointly
      if show_micro_average:  # If the flag is set to True
          precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_binarized.ravel(),
                                                                          pred.ravel())
          average_precision["micro"] = average_precision_score(y_test_binarized, pred, average="micro")
          print('Average precision score, micro-averaged over all classes: {0:0.2f}'
                .format(average_precision["micro"]))

      plt.figure()
      if show_micro_average:  # If the flag is set to True
          plt.step(recall['micro'], precision['micro'], where='post')
          plt.title(
              'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
              .format(average_precision["micro"]))

      plt.xlabel('Recall')
      plt.ylabel('Precision')
      plt.ylim([0.0, 1.05])
      plt.xlim([0.0, 1.0])

      for i in show_classes:
          plt.plot(recall[i], precision[i], label='{0} (AP = {1:0.2f})'.format(reverse_bird_dics[i], average_precision[i]))

    plt.legend(loc="best")
    plt.show()
    
    
    # compare different model with the same class
    def compare_roc_curves(models_output, true_labels, bird_name):
    """
    models_output: a dictionary that contains model names as keys and predicted probabilities as values
    true_labels: true labels (ground truth)
    bird_name: the name of the bird species you want to plot the ROC curve for
    bird_dict: dictionary that maps bird names to indices
    """
    # Create dic for birds name mapping
    bird_dic = {'bermuda_petrel':0,
    'black_winged_petrel':1,
    'cooks_petrel':2,
    'feas_petrel':3,
    'galapagos_petrel':4,
    'goulds_petrel':5,
    'hawaiian_petrel':6,
    'masatierra_petrel':7,
    'soft_plumaged_petrel':8}

    # Initialize a plot
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')

    # Store the model names, AUC scores, and ROC curve data
    legend_data = []

    # Calculate and plot the ROC curve for each model
    for model_name, y_pred_prob in models_output.items():
        # Convert the true labels to a binary format for this species
        true_binary = [1 if label == bird_name else 0 for label in true_labels]

        # Get the index for the bird species
        bird_index = bird_dic[bird_name]

        # Calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(true_binary, y_pred_prob[:, bird_index])

        # Calculate the AUC
        auc_score = roc_auc_score(true_binary, y_pred_prob[:, bird_index])

        # Store the model name, AUC score, and ROC curve data for later use
        legend_data.append((model_name, auc_score, fpr, tpr))

    # Sort the legend data in descending order based on AUC score
    legend_data.sort(key=lambda x: x[1], reverse=True)

    # Add each ROC curve to the plot
    for model_name, auc_score, fpr, tpr in legend_data:
        plt.plot(fpr, tpr, label=f'{model_name} (area = {auc_score:.2f})')

    # Add labels and a legend to the plot
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'{bird_name} ROC')
    plt.legend(loc='best')
    plt.show()
    
    
    
    # create chart and save it for later model experiment
    def create_chart(history, pred_labels, model, true, save=True, csv_path='/Users/aristo/Desktop/final dissertation', csv_name='model', report_name='model'):

    # Create dic for birds name mapping
    bird_dics = {'bermuda_petrel':0,
    'black_winged_petrel':1,
    'cooks_petrel':2,
    'feas_petrel':3,
    'galapagos_petrel':4,
    'goulds_petrel':5,
    'hawaiian_petrel':6,
    'masatierra_petrel':7,
    'soft_plumaged_petrel':8}

    pred = [bird_dics[bird] for bird in pred_labels]
    true = [bird_dics[bird] for bird in true]

    # Reverse dictionary for later use
    reverse_bird_dics = {v: k for k, v in bird_dics.items()}

    # train_val loss
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # train_val accuracy
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # test accuracy
    steps = len(test_df) // BATCH_SIZE
    test_accuracy = model.evaluate(test_images, steps=steps)

    # epochs
    epochs = np.arange(1, len(train_loss) + 1)

    # confusion matrix report
    report = classification_report(true, pred, output_dict=True)
    d = pd.DataFrame(report).transpose()

    # create dataframe
    df = pd.DataFrame({'epochs':epochs,
                     'train_loss':train_loss,
                     'val_loss':val_loss,
                     'train_accuracy':train_accuracy,
                     'val_accuracy':val_accuracy})

    # Create a confusion matrix
    cm = confusion_matrix(true, pred)
    cm_df = pd.DataFrame(cm)

    # Calculate the correct and incorrect predictions per class
    correct_predictions = np.diag(cm)
    incorrect_predictions = cm.sum(axis=1) - correct_predictions
    total_images = cm.sum(axis=1)
    correct_percentage = (correct_predictions / total_images) * 100
    incorrect_percentage = 100 - correct_percentage

    # Create a dataframe for correct, incorrect predictions and percentage
    pred_df = pd.DataFrame({
        'bird_names': [reverse_bird_dics[i] for i in range(len(reverse_bird_dics))],
        'total_test_images': total_images,
        'correct_percentage': correct_percentage.round(1),
        'incorrect_percentage': incorrect_percentage.round(1),
    })

    # Sort the dataframe by correct_percentage in descending order
    pred_df = pred_df.sort_values(by='correct_percentage', ascending=False)

    # save to csv
    if save:
        df.to_csv(csv_path + csv_name + '.csv', index=False, sep=',')
        d.to_csv(csv_path + csv_name + ' report' + '.csv', sep=',')
        pred_df.to_csv(csv_path + csv_name + ' predictions' + '.csv', sep=',')

    return test_accuracy, df, pred_df

