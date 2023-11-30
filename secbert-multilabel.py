
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

from sklearn.metrics import *
from sklearn.model_selection import train_test_split

from transformers import BertForSequenceClassification, BertTokenizerFast
from dscv.utils.save_report import save_classification, calculate_score
from dscv.datasets import OpcodeData

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


def freeze_k_layer(secBertClassifier, k=1):
  for param in secBertClassifier.bert.encoder.layer[0:k].parameters():
    param.requires_grad = False

class CustomClassifier(nn.Module):
    def __init__(self, original_model):
        super(CustomClassifier, self).__init__()
        self.original_model = original_model
        self.activation = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.original_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        logits = self.activation(outputs.logits)
        return logits

def train_steps(training_loader, model, loss_f, optimizer):
    print('Training...')
    training_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    train_acc = 0.
    train_f1 = 0.

    model.train()

    for step, batch in enumerate(training_loader):
        # push the batch to gpu
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        targets = batch['targets'].to(device)

        preds = model(ids, attention_mask=mask, token_type_ids=token_type_ids)

        loss = loss_f(preds, targets)
        training_loss += loss.item()

        preds = preds.detach().cpu().numpy()
        preds = np.where(preds>=0.5, 1, 0)
        label_ids = targets.to('cpu').numpy()

        acc_score, F1_score = calculate_score(label_ids, preds)
        train_acc += acc_score
        train_f1 += F1_score
        nb_tr_steps += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # # When using GPU
        optimizer.step()

    epoch_loss = training_loss / nb_tr_steps
    epoch_acc = train_acc / nb_tr_steps
    epoch_f1 = train_f1 / nb_tr_steps
    return epoch_loss, epoch_acc, epoch_f1


def evaluate_steps(validating_loader, model, loss_f):
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []
    total_labels = []
    # iterate over batches
    for step, batch in enumerate(validating_loader):
        # push the batch to gpu
        b_input_ids = batch['ids'].to(device)
        b_input_mask = batch['mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        b_labels = batch['targets'].to(device)

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(b_input_ids, attention_mask=b_input_mask, token_type_ids=token_type_ids)

            # compute the validation loss between actual and predicted values
            loss = loss_f(preds, b_labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()
            preds = np.where(preds>=0.5, 1, 0)
            total_preds += list(preds)
            total_labels += b_labels.tolist()
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(validating_loader)
    acc_score, F1_score = calculate_score(total_labels, total_preds)

    return avg_loss, acc_score, F1_score


def predict(testing_loader, model):
    print("\Predicting...")
    # deactivate dropout layers
    model.eval()

    # empty list to save the model predictions
    total_preds = []
    total_labels = []
    # iterate over batches
    for step, batch in enumerate(testing_loader):
        # push the batch to gpu
        b_input_ids = batch['ids'].to(device)
        b_input_mask = batch['mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        b_labels = batch['targets'].to(device)

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(b_input_ids, attention_mask=b_input_mask, token_type_ids=token_type_ids)

            preds = preds.detach().cpu().numpy()
            preds = np.where(preds>=0.5, 1, 0)
            total_preds += list(preds)
            total_labels += b_labels.tolist()

    return total_preds, total_labels


def train(epochs, model, optimizer, criterion, dataloader):
  data_train_loader, data_val_loader = dataloader
  # set initial loss to infinite
  best_valid_loss = float('inf')
  train_losses = []
  valid_losses = []
  train_accuracies = []
  valid_accuracies = []

  if os.path.isdir('./trained') == False:
    os.mkdir('./trained')

  for epoch in range(epochs):
    print('Epoch {}/{} \t'.format(epoch + 1, epochs))
    start_time = time.time()
    train_loss, train_acc, _ = train_steps(data_train_loader, model, criterion, optimizer)

    valid_loss, valid_acc, _ = evaluate_steps(data_val_loader, model, criterion)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'secbert-multilabel.pt')
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_acc)
    valid_accuracies.append(valid_acc)

    elapsed_time = time.time() - start_time

    print('loss={:.4f} \t accuracy={:.4f} \t val_loss={:.4f}  \t val_acc={:.4f}  \t time={:.2f}s'.format(train_loss, train_acc, valid_loss, valid_acc, elapsed_time))
  return train_accuracies, valid_accuracies, train_losses, valid_losses

def plot_graph(epochs, train, valid, tittle):
    fig = plt.figure(figsize=(12,12))
    plt.title(tittle)
    plt.plot(list(np.arange(epochs) + 1) , train, label='train')
    plt.plot(list(np.arange(epochs) + 1), valid, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')

def run(k=1):
    # Defining some key variables that will be used later on in the training
    max_length = 512
    TRAIN_BATCH_SIZE = 128
    VALID_BATCH_SIZE = 128
    epochs = 10
    LEARNING_RATE = 1e-05
    num_class = 4

    """### Load pretrained model form huggingface"""
    secBertTokenizer = BertTokenizerFast.from_pretrained("jackaduma/SecBERT", do_lower_case=True)
    secBertClassifier = BertForSequenceClassification.from_pretrained("jackaduma/SecBERT", num_labels = num_class)
    freeze_k_layer(secBertClassifier=secBertClassifier, k=k)
    
    """Custom activation"""
    secBertClassifierMultilabel = CustomClassifier(secBertClassifier)
    """### Read data"""
    data_folder = os.getcwd() + '/data/'
    data = pd.read_csv(data_folder + '/Data_Cleansing.csv')
    selected_columns = ['BYTECODE', 'Timestamp dependence', 'Outdated Solidity version', 'Frozen Ether', 'Delegatecall Injection']
    data = data.loc[:, selected_columns]

    labels = data.iloc[:, -4:].keys().tolist()
    test = data.iloc[:, -4:].to_numpy()
    values = np.sum(test, axis=0)

    print(dict(zip(labels, values)))

    X, y = data['BYTECODE'].to_numpy(), data.iloc[:, -4:].to_numpy()

    """Split data"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    """
    Create dataset
    """
    training_set = OpcodeData(X_train, y_train, secBertTokenizer, max_length)
    validating_set = OpcodeData(X_val, y_val, secBertTokenizer, max_length)
    testing_set = OpcodeData(X_test, y_test, secBertTokenizer, max_length)

    """
    Create data loader
    """
    training_loader = DataLoader(training_set, batch_size=TRAIN_BATCH_SIZE)
    validating_loader = DataLoader(validating_set, batch_size=VALID_BATCH_SIZE)
    testing_loader = DataLoader(testing_set, batch_size=VALID_BATCH_SIZE)
    secBertClassifierMultilabel.to(device)

    """
    Define loss function and optimizer
    """
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(params = secBertClassifierMultilabel.parameters(), lr=LEARNING_RATE)

    """
    Train model
    """
    train_accuracies, valid_accuracies, train_losses, valid_losses = train(epochs, secBertClassifierMultilabel, optimizer, criterion, (training_loader, validating_loader))

    """
    Plot the result of training process
    """
    plot_graph(epochs, train_losses, valid_losses, "Train/Validation Loss")
    plot_graph(epochs, train_accuracies, valid_accuracies, "Train/Validation Accuracy")

    """
    Evaluate model on test set and save the result
    """
    y_preds, total_test = predict(testing_loader, secBertClassifierMultilabel)
    save_classification(y_test=np.array(total_test), y_pred=np.array(y_preds), labels=labels, out_dir='secbert-multilabel-freeze-'+k+'-layer.csv')
if __name__ == '__main__':
    run(k=5)
    run(k=4)
    run(k=3)
    run(k=2)
    run(k=1)
    run(k=6)