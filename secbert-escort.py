import pandas as pd
import os
import collections
import numpy as np
import zipfile
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast

from sklearn.metrics import *
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
 dev = "cuda:0"
else:
 dev = "cpu"
device = torch.device(dev)
device

def freeze_k_layer(secBert, k=1):
  for param in secBert.encoder.layer[0:k].parameters():
    param.requires_grad = False

class Branch(nn.Module):
  def __init__(self, input_size, hidden1_size, hidden2_size, dropout, num_outputs):
    super(Branch, self).__init__()

    self.dense1 = nn.Linear(input_size, hidden1_size)
    self.dropout = nn.Dropout(p=dropout)
    self.dense2 = nn.Linear(hidden1_size, hidden2_size)
    self.dense3 = nn.Linear(hidden2_size, num_outputs)

  def forward(self, x):
    out_dense1 = self.dense1(x)
    out_dropout = self.dropout(out_dense1)
    out_dense2 = self.dense2(out_dropout)
    out_dense3 = self.dense3(out_dense2)

    return out_dense3

class BaseModel(nn.Module):
    def __init__(self, original_model, num_classes):
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
        self.original_model = original_model
        self.branches = nn.ModuleList([Branch(768, 128, 64, 0.1, 2) for _ in range(num_classes)])
        self.activation = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        out_bert = self.original_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooler_out = out_bert.pooler_output
        output_branches = [branch(pooler_out) for branch in self.branches]
        outputs = [self.activation(out_branch) for out_branch in output_branches]

        # apply softmax function for each branch
        out_soft = [self.activation(out) for out in outputs]
        out_soft_max_indices = [torch.argmax(out, dim=1) for out in out_soft]
        out_soft_max_indices = torch.stack(out_soft_max_indices, dim=1)

        return out_soft, out_soft_max_indices

"""### Preprocess data"""
class OpcodeData(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.X = X
        self.targets = y
        self.max_len = max_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        text = str(self.X[index])

        inputs = self.tokenizer(
            text,
            None,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'index': index,
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

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

"""### Create model and fine-tuning"""
def calculate_score(y_true, preds):
    acc_score = accuracy_score(y_true, preds)

    return acc_score

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

def train_steps(training_loader, model, loss_f, optimizer):
    print('Training...')
    training_loss = 0
    nb_tr_steps = 0
    train_acc = 0.
    misclassify_train_data = {}

    model.train()

    for step, batch in enumerate(training_loader):
        # push the batch to gpu
        indices = batch['index'].numpy()
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        targets = batch['targets'].to(device)

        preds, max_indices = model(ids, attention_mask=mask, token_type_ids=token_type_ids)

        # calculate the loss for each branch
        losses = [loss_f(preds[i], targets[:, i]) for i in range(targets.shape[1])]
        average_loss = sum(losses) / targets.shape[1]
        training_loss += average_loss.item()

        label_ids = targets.to('cpu').numpy()
        max_indices = max_indices.detach().cpu().numpy()
        acc_score = accuracy_score(label_ids, max_indices)
        train_acc += acc_score

        misclassify_data = get_misclassified_data(label_ids, max_indices, indices)
        misclassify_train_data.update(misclassify_data)

        nb_tr_steps += 1

        optimizer.zero_grad()
        average_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # When using GPU
        optimizer.step()

    epoch_loss = training_loss / nb_tr_steps
    epoch_acc = train_acc / nb_tr_steps

    return epoch_loss, epoch_acc, misclassify_train_data


def evaluate_steps(validating_loader, model, loss_f):
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss = 0

    # empty list to save the model predictions
    total_preds = []
    total_labels = []
    # iterate over batches
    for step, batch in enumerate(validating_loader):
        # push the batch to gpu
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        targets = batch['targets'].to(device)

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds, max_indices = model(ids, attention_mask=mask, token_type_ids=token_type_ids)

            # compute the validation loss between actual and predicted values
            losses = [loss_f(preds[i], targets[:, i]) for i in range(targets.shape[1])]
            average_loss = sum(losses) / targets.shape[1]
            total_loss += average_loss.item()

            max_indices = max_indices.detach().cpu().numpy()
            total_preds += list(max_indices)
            total_labels += targets.tolist()
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(validating_loader)
    acc_score = accuracy_score(total_labels, total_preds)

    return avg_loss, acc_score

def predict(testing_loader, model):
    print("\nPredicting...")
    # deactivate dropout layers
    model.eval()

    # empty list to save the model predictions
    total_preds = []
    total_labels = []
    # iterate over batches
    for step, batch in enumerate(testing_loader):
        # push the batch to gpu
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        targets = batch['targets'].to(device)

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds, max_indices = model(ids, attention_mask=mask, token_type_ids=token_type_ids)

            max_indices = max_indices.detach().cpu().numpy()
            total_preds += list(max_indices)
            total_labels += targets.tolist()

    return total_labels, total_preds


def train(epochs, model, optimizer, criterion, dataloader):
  data_train_loader, data_val_loader = dataloader
  # set initial loss to infinite
  best_valid_loss = float('inf')
  train_losses = []
  valid_losses = []
  train_accuracies = []
  valid_accuracies = []
  misclassify_train_data = {}

  for epoch in range(epochs):
    print('Epoch {}/{} '.format(epoch + 1, epochs))
    start_time = time.time()
    train_loss, train_acc, misclassify_train_steps_data = train_steps(data_train_loader, model, criterion, optimizer)
    valid_loss, valid_acc = evaluate_steps(data_val_loader, model, criterion)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'secbert-escort.pt')
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_acc)
    valid_accuracies.append(valid_acc)
    misclassify_train_data.update(misclassify_train_steps_data)

    elapsed_time = time.time() - start_time

    print('\t loss={:.4f} \t accuracy={:.4f} \t val_loss={:.4f}  \t val_acc={:.4f}  \t time={:.2f}s'.format(train_loss, train_acc, valid_loss, valid_acc, elapsed_time))

  return train_accuracies, valid_accuracies, train_losses, valid_losses, misclassify_train_data

def plot_graph(epochs, train, valid, tittle):
    fig = plt.figure(figsize=(12,12))
    plt.title(tittle)
    plt.plot(list(np.arange(epochs) + 1) , train, label='train')
    plt.plot(list(np.arange(epochs) + 1), valid, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')

def run(k=6):
  # Defining some key variables that will be used later on in the training
  data_folder = os.getcwd() + '/data-multilabel/'
  labels = ['Timestamp dependence', 'Outdated Solidity version', 'Frozen Ether', 'Delegatecall Injection']
  max_length = 5500
  TRAIN_BATCH_SIZE = 32
  VALID_BATCH_SIZE = 32
  EPOCHS = 5
  LEARNING_RATE = 1e-04
  num_class = 4

  # Read data
  X_train = pd.read_csv(data_folder+'X_train.csv').to_numpy()
  X_test = pd.read_csv(data_folder+'X_test.csv').to_numpy()
  X_val = pd.read_csv(data_folder+'X_val.csv').to_numpy()

  y_train = pd.read_csv(data_folder+'y_train.csv').to_numpy()
  y_test = pd.read_csv(data_folder+'y_test.csv').to_numpy()
  y_val = pd.read_csv(data_folder+'y_val.csv').to_numpy()

  # Load pretrained model form huggingface
  secBertTokenizer = BertTokenizerFast.from_pretrained("jackaduma/SecBERT", do_lower_case=True)
  secBertModel = BertModel.from_pretrained("jackaduma/SecBERT", num_labels = num_class)

  # freeze for fine tuning
  freeze_k_layer(secBertModel, k=k)
  
  # create a custom model from the secbert model
  secBertClassifierMultilabel = BaseModel(original_model=secBertModel, num_classes=num_class)
  print(secBertClassifierMultilabel)
  secBertClassifierMultilabel.to(device)

  # create dataset
  training_set = OpcodeData(X_train, y_train, secBertTokenizer, max_length)
  validating_set = OpcodeData(X_val, y_val, secBertTokenizer, max_length)
  testing_set = OpcodeData(X_test, y_test, secBertTokenizer, max_length)

  # Create generator for Dataset with BATCH_SIZE
  training_loader = DataLoader(training_set, batch_size=TRAIN_BATCH_SIZE)
  validating_loader = DataLoader(validating_set, batch_size=VALID_BATCH_SIZE)
  testing_loader = DataLoader(testing_set, batch_size=VALID_BATCH_SIZE)

  # Creating the loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(params = secBertClassifierMultilabel.parameters(), lr=LEARNING_RATE)

  # train the model
  train_accuracies, valid_accuracies, train_losses, valid_losses, misclassify_train_data = train(EPOCHS, secBertClassifierMultilabel, optimizer, criterion, (training_loader, validating_loader))
  
  df = pd.DataFrame.from_dict(misclassify_train_data, orient='index', columns=labels)
  df.index.name = 'X_train_index'
  df.to_csv(data_folder+'misclassified-data.csv')

  # Plot the result of training process
  plot_graph(EPOCHS, train_losses, valid_losses, "Train/Validation Loss")
  plot_graph(EPOCHS, train_accuracies, valid_accuracies, "Train/Validation Accuracy")

  # Evaluate model on test set and save the result
  total_labels, total_preds = predict(testing_loader, secBertClassifierMultilabel)
  df_labels = pd.DataFrame(total_labels, columns=labels)
  df_preds = pd.DataFrame(total_preds, columns=labels)

  df_labels.to_csv(data_folder+'labels-test-secbert-escort.csv')
  df_preds.to_csv(data_folder+'preds-secbert-escort.csv')

  # Save the result
  save_classification(y_test=np.array(total_labels), y_pred=np.array(total_preds), labels=labels, out_dir='escort-secbert.csv')

if __name__ == '__main__':
  run(k=6)