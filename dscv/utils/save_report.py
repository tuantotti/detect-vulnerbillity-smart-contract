import numpy as np
import pandas as pd
from sklearn.metrics import *
import os
import matplotlib.pyplot as plt

def save_classification(y_test, y_pred, out_dir, labels):
  if isinstance(y_pred, np.ndarray) == False:
    y_pred = y_pred.toarray()
  
  def accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        numerator = sum(np.logical_and(y_true[i], y_pred[i]))
        denominator = sum(np.logical_or(y_true[i], y_pred[i]))
        if denominator != 0:
          temp += numerator / denominator
    return temp / y_true.shape[0]

  out = classification_report(y_test,y_pred, output_dict=True, target_names=labels)
  total_support = out['samples avg']['support']

  mr = accuracy_score(y_test, y_pred)
  acc = accuracy(y_test,y_pred)
  hm = hamming_loss(y_test, y_pred)

  out['Exact Match Ratio'] = {'precision': mr, 'recall': mr, 'f1-score': mr, 'support': total_support}
  out['Hamming Loss'] = {'precision': hm, 'recall': hm, 'f1-score': hm, 'support': total_support}
  out['Accuracy'] = {'precision': acc, 'recall': acc, 'f1-score': acc, 'support': total_support}
  out_df = pd.DataFrame(out).transpose()
  print(out_df)

  out_df.to_csv(out_dir)

  return out_df

def plot_graph(epochs, train, valid, tittle):
    fig = plt.figure(figsize=(12,12))
    plt.title(tittle)
    plt.plot(list(np.arange(epochs) + 1) , train, label='train')
    plt.plot(list(np.arange(epochs) + 1), valid, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')
  

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


def wisdomnet_classification_report(y_true, y_pred, labels):
  cfm = wisdomnet_cfm(y_true, y_pred)
  num_labels = len(labels)
  sum_all_TP, sum_all_FP, sum_all_FN = 0, 0, 0
  sum_pre, sum_re, sum_f1 = 0., 0., 0.

  report = {}

  for i in range(num_labels):
    TN = cfm[i, 0, 0]
    FN = cfm[i, 1, 0]
    TP = cfm[i, 1, 1]
    FP = cfm[i, 0, 1]
    reject_labels = cfm[i, 0, 2]

    label = labels[i]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * precision * recall) / (precision + recall)

    sum_pre += precision
    sum_re += recall
    sum_f1 += F1

    sum_all_TP += TP
    sum_all_FP += FP
    sum_all_FN += FN

    report[label] = {
        'Precision': precision,
        'Recall': recall,
        'F1': F1,
        'Reject Label': reject_labels
    }

  micro_p = sum_all_TP / (sum_all_TP + sum_all_FP)
  micro_r = sum_all_TP / (sum_all_TP + sum_all_FN)
  micro_f = (2 * micro_p * micro_r) / (micro_r + micro_p)

  report['micro avg'] = {
    'Precision': micro_p,
    'Recall': micro_r,
    'F1': micro_f,
    'Reject Label': 0
  }

  report['macro avg'] = {
    'Precision': sum_pre / num_labels,
    'Recall': sum_re / num_labels,
    'F1': sum_f1 / num_labels,
    'Reject Label': 0
  }

  acc = calculate_accuracy(y_true, y_pred)
  report['accuracy'] = {
    'Precision': acc,
    'Recall': acc,
    'F1': acc,
    'Reject Label': 0
  }

  df = pd.DataFrame(report).T
  df.columns = ['Precision', 'Recall', 'F1', 'Reject Label']

  return df