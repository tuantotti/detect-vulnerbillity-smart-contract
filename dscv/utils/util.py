import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt

def get_misclassified_data(labels, preds, indices):
  misclassify_data = {}
  for i in range(len(labels)):
    is_append = False
    reject_label = np.array(labels[i])
    for j in range(len(labels[i])):
      if labels[i, j] != preds[i, j]:
        reject_label[j] = 2 # reject label
        is_append = True

    if is_append:
      x_train_index = indices[i]
      misclassify_data[x_train_index] = np.array(reject_label)
  return misclassify_data


def freeze_k_layer(secBert, k=1):
  for param in secBert.encoder.layer[0:k].parameters():
    param.requires_grad = False
    

def wisdomnet_cfm(y_true, y_pred, reject_label=2):

  if y_true.shape[0] != y_pred.shape[0] or y_true.shape[1] != y_pred.shape[1]:
    return Exception("Shape is not equally")

  num_samples = y_true.shape[0]
  num_label = y_true.shape[1]
  cfm = np.zeros(shape=(num_label, 2, 3), dtype=int)

  for i in range(num_label):
    num_reject = 0
    TN, FN, TP, FP = 0, 0, 0, 0
    for j in range(num_samples):
      if y_pred[j][i] == reject_label:
        num_reject+=1
      elif y_true[j][i] == 1 and y_pred[j][i] == 1:
        TP += 1  # True Positive
      elif y_true[j][i] == 0 and y_pred[j][i] == 1:
        FP += 1  # False Positive
      elif y_true[j][i] == 0 and y_pred[j][i] == 0:
        TN += 1  # True Negative
      elif y_true[j][i] == 1 and y_pred[j][i] == 0:
        FN += 1  # False Negative

    cfm[i, 0, 0] = TN
    cfm[i, 1, 0] = FN
    cfm[i, 1, 1] = TP
    cfm[i, 0, 1] = FP
    cfm[i, 0, 2] = num_reject
    cfm[i, 1, 2] = num_reject

  return cfm

def cal_wisdomnet_acc(y_true, y_pred):
  if y_true.shape[0] != y_pred.shape[0] or y_true.shape[1] != y_pred.shape[1]:
    return Exception("Shape is not equally")

  num_samples = y_true.shape[0]
  num_labels = y_true.shape[1]
  acc_score = 0.

  for i in range(num_samples):
    num_acc = 0
    total = 0
    for j in range(num_labels):
      if y_true[i, j] == y_pred[i, j] and y_true[i, j] != 0:
        num_acc+=1

      if y_true[i, j] != 0 or y_pred[i, j] != 0:
        total += 1
    local_acc = 0
    if total != 0:
        local_acc = num_acc / total
    acc_score += local_acc
  return acc_score / num_samples

    
def calculate_score(y_true, preds):
    acc_score = accuracy_score(y_true, preds)
    return acc_score
  

def plot_graph(epochs, train, valid, tittle):
    fig = plt.figure(figsize=(12,12))
    plt.title(tittle)
    plt.plot(list(np.arange(epochs) + 1) , train, label='train')
    plt.plot(list(np.arange(epochs) + 1), valid, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')