import numpy as np
import pandas as pd
from sklearn.metrics import *
import os

def save_classification(y_test, y_pred, out_dir, labels):
  if os.path.isdir('./report') == False:
    os.mkdir('./report')
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