import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import *
from dscv.utils.feature_extraction_utils import BagOfWord, TfIdf
from dscv.utils.process_text import Tokenizer
from dscv.datasets import EscortOpcodeData
from dscv.models.models import Escort
from dscv.utils.save_report import save_classification


if torch.cuda.is_available():
 dev = "cuda:0"
else:
 dev = "cpu"
device = torch.device(dev)
device


def train_steps(training_loader, model, loss_f, optimizer):
    training_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    train_acc = 0.

    model.train()
    for step, batch in enumerate(training_loader):
        # push the batch to gpu
        inputs = batch['ids'].to(device)
        labels = batch['targets'].to(device)
        tfidf_inputs = batch['tfidf'].to(device)

        preds = model(inputs, tfidf_inputs)

        loss = loss_f(preds, labels)
        training_loss += loss.item()

        preds = preds.detach().cpu().numpy()
        preds = np.where(preds>=0.5, 1, 0)
        labels = labels.to('cpu').numpy()

        acc_score = calculate_score(labels, preds)
        train_acc += acc_score

        nb_tr_steps += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # When using GPU
        optimizer.step()

    epoch_loss = training_loss / nb_tr_steps
    epoch_acc = train_acc / nb_tr_steps
    return epoch_loss, epoch_acc

def evaluate_steps(validating_loader, model, loss_f):
    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []
    total_labels = []
    # iterate over batches
    for step, batch in enumerate(validating_loader):
        # push the batch to gpu
        inputs = batch['ids'].to(device)
        labels = batch['targets'].to(device)
        tfidf_inputs = batch['tfidf'].to(device)


        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(inputs, tfidf_inputs)

            # compute the validation loss between actual and predicted values
            loss = loss_f(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()
            preds = np.where(preds>=0.5, 1, 0)
            total_preds += list(preds)
            total_labels += labels.tolist()
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(validating_loader)
    acc_score = calculate_score(total_labels, total_preds)

    return avg_loss, acc_score

"""### Training loop"""

def train(epochs, model, optimizer, criterion):
  # empty lists to store training and validation loss of each epoch
  # set initial loss to infinite
  best_valid_loss = float('inf')
  train_losses = []
  valid_losses = []
  train_accuracies = []
  valid_accuracies = []

  for epoch in range(epochs):
    start_time = time.time()
    train_loss, train_acc = train_steps(data_train_loader, model, criterion, optimizer)

    valid_loss, valid_acc = evaluate_steps(data_val_loader, model, criterion)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './trained/escort-bow-gru.pt')
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_acc)
    valid_accuracies.append(valid_acc)

    elapsed_time = time.time() - start_time

    print('Epoch {}/{} \t loss={:.4f} \t accuracy={:.4f} \t val_loss={:.4f}  \t val_acc={:.4f}  \t time={:.2f}s'.format(epoch + 1, epochs, train_loss, train_acc, valid_loss, valid_acc, elapsed_time))
  return train_accuracies, valid_accuracies, train_losses, valid_losses

def plot_graph(epochs, train, valid, tittle):
    fig = plt.figure(figsize=(12,12))
    plt.title(tittle)
    plt.plot(list(np.arange(epochs) + 1) , train, label='train')
    plt.plot(list(np.arange(epochs) + 1), valid, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')

"""## Test Model"""

def predict(testing_loader, model, loss_f):
    # deactivate dropout layers
    model.eval()

    # empty list to save the model predictions
    total_preds = []
    total_labels = []
    start_time = time.time()
    # iterate over batches
    for step, batch in enumerate(testing_loader):
        # push the batch to gpu
        inputs = batch['ids'].to(device)
        labels = batch['targets'].to(device)
        tfidf_inputs = batch['tfidf'].to(device)

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(inputs, tfidf_inputs)

            preds = preds.detach().cpu().numpy()
            preds = np.where(preds>=0.5, 1, 0)
            total_preds += list(preds)
            total_labels += labels.tolist()

    execution_time = (time.time() - start_time) / len(total_labels)
    return total_preds, total_labels, execution_time

"""## Run"""

epochs = 20
EMBEDDED_SIZE = 5
NUM_OUTPUT_NODES = 4
NUM_LAYERS = 1
DROPOUT = 0.2
input_size = 4100
batch_size = 128
GRU_HIDDEN_SIZE = 256
data_folder = os.getcwd()+'/data/'

X_train = pd.read_csv(data_folder+'X_train.csv')['BYTECODE'].to_numpy()
X_test = pd.read_csv(data_folder+'X_test.csv')['BYTECODE'].to_numpy()
X_val = pd.read_csv(data_folder+'X_val.csv')['BYTECODE'].to_numpy()

y_train = pd.read_csv(data_folder+'y_train.csv').to_numpy()
y_test = pd.read_csv(data_folder+'y_test.csv').to_numpy()
y_val = pd.read_csv(data_folder+'y_val.csv').to_numpy()


tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts=X_train)
SIZE_OF_VOCAB = len(tokenizer.word_index.keys())


train_dataset = EscortOpcodeData(X_train, y_train, tokenizer, input_size)
val_dataset = EscortOpcodeData(X_val, y_val, tokenizer, input_size)
test_dataset = EscortOpcodeData(X_test, y_test, tokenizer, input_size)

data_train_loader = DataLoader(train_dataset, batch_size=batch_size)
data_val_loader = DataLoader(val_dataset, batch_size=batch_size)
data_test_loader = DataLoader(test_dataset, batch_size=batch_size)

model = Escort(vocab_size=SIZE_OF_VOCAB, embedd_size=EMBEDDED_SIZE,
               gru_hidden_size=GRU_HIDDEN_SIZE, n_layers=NUM_LAYERS,
               num_classes=NUM_OUTPUT_NODES)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.BCELoss()

train_accuracies, valid_accuracies, train_losses, valid_losses = train(epochs, model, optimizer, criterion)

plot_graph(epochs, train_losses, valid_losses, "Train/Validation Loss")
plot_graph(epochs, train_accuracies, valid_accuracies, "Train/Validation Accuracy")

total_preds, total_labels, execution_time = predict(data_test_loader, model, criterion)

print(execution_time)

labels = ['Timestamp dependence', 'Outdated Solidity version', 'Frozen Ether', 'Delegatecall Injection']
save_classification(y_pred=np.array(total_preds), y_test=np.array(total_labels), labels=labels, out_dir='escort-bow-gru.csv')
