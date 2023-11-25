import numpy as np

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