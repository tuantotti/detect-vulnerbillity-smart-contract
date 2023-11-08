import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from utils.save_report import save_classification
from utils.process_text import Tokenizer, pad_sequences

if torch.cuda.is_available():
 dev = "cuda:0"
else:
 dev = "cpu"
device = torch.device(dev)
print(device)
"""## Create Model"""
class Branch(nn.Module):
  def __init__(self, INPUT_SIZE, hidden1_size, hidden2_size, dropout, num_outputs):
    super(Branch, self).__init__()

    self.dense1 = nn.Linear(INPUT_SIZE, hidden1_size)
    self.dense2 = nn.Linear(hidden1_size, hidden2_size)
    self.dense3 = nn.Linear(hidden2_size, num_outputs)
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, x):
    out_dense1 = self.dense1(x)
    out_dropout = self.dropout(out_dense1)
    out_dense2 = self.dense2(out_dropout)
    out_dense3 = self.dense3(out_dense2)

    return out_dense3

class Escort(nn.Module):
  def __init__(self, vocab_size, embedd_size, gru_hidden_size, n_layers, num_classes):
    super(Escort, self).__init__()
    self.word_embeddings = nn.Embedding(vocab_size, embedd_size)
    self.gru = nn.GRU(embedd_size, gru_hidden_size, num_layers=n_layers)
    self.branches = nn.ModuleList([Branch(gru_hidden_size, 128, 64, 0.2, 1) for _ in range(num_classes)])
    self.sigmoid = nn.Sigmoid()

  def forward(self, sequence):
    embeds = self.word_embeddings(sequence)
    gru_out, _ = self.gru(embeds)
    output_branches = [branch(gru_out[:, -1, :]) for branch in self.branches]
    output_branches = torch.cat(output_branches, dim=1)
    # outputs = self.sigmoid(output_branches)
    return outputs

"""## Train model"""
"""### Train and Validation Steps"""

def calculate_score(y_true, preds):
    acc_score = accuracy_score(y_true, preds)

    return acc_score

def train_steps(training_loader, model, loss_f, optimizer):
    training_loss = 0
    nb_tr_steps = 0
    train_acc = 0.

    model.train()
    for step, batch in enumerate(training_loader):
        # push the batch to gpu
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        preds = model(inputs)

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

    total_loss = 0

    # empty list to save the model predictions
    total_preds = []
    total_labels = []
    # iterate over batches
    for step, batch in enumerate(validating_loader):
        # push the batch to gpu
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(inputs)

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
def train(EPOCHS, model, optimizer, criterion, dataloader):
  data_train_loader, data_val_loader = dataloader
  # empty lists to store training and validation loss of each epoch
  # set initial loss to infinite
  best_valid_loss = float('inf')
  train_losses = []
  valid_losses = []
  train_accuracies = []
  valid_accuracies = []

  for epoch in range(EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train_steps(data_train_loader, model, criterion, optimizer)

    valid_loss, valid_acc = evaluate_steps(data_val_loader, model, criterion)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './trained/escort.pt')
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_acc)
    valid_accuracies.append(valid_acc)

    elapsed_time = time.time() - start_time

    print('Epoch {}/{} \t loss={:.4f} \t accuracy={:.4f} \t val_loss={:.4f}  \t val_acc={:.4f}  \t time={:.2f}s'.format(epoch + 1, EPOCHS, train_loss, train_acc, valid_loss, valid_acc, elapsed_time))
  return train_accuracies, valid_accuracies, train_losses, valid_losses

def plot_graph(EPOCHS, train, valid, tittle):
    fig = plt.figure(figsize=(12,12))
    plt.title(tittle)
    plt.plot(list(np.arange(EPOCHS) + 1) , train, label='train')
    plt.plot(list(np.arange(EPOCHS) + 1), valid, label='validation')
    plt.xlabel('num_EPOCHS', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')

"""## Test Model"""

def predict(testing_loader, model):
    # deactivate dropout layers
    model.eval()

    # empty list to save the model predictions
    total_preds = []
    total_labels = []
    start_time = time.time()
    # iterate over batches
    for step, batch in enumerate(testing_loader):
        # push the batch to gpu
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(inputs)

            preds = preds.detach().cpu().numpy()
            preds = np.where(preds>=0.5, 1, 0)
            total_preds += list(preds)
            total_labels += labels.tolist()

    execution_time = (time.time() - start_time) / len(total_labels)
    return total_preds, total_labels, execution_time

def run():
    data_folder = os.getcwd() + '/data-multilabel/'
    data = pd.read_csv(data_folder + 'Data_Cleansing.csv')
    selected_columns = ['BYTECODE', 'Timestamp dependence', 'Outdated Solidity version', 'Frozen Ether', 'Delegatecall Injection']
    data = data.loc[:, selected_columns]
    labels = data.iloc[:, -4:].keys().tolist()
    test = data.iloc[:, -4:].to_numpy()
    values = np.sum(test, axis=0)
    print(dict(zip(labels, values)))
    X, y = data['BYTECODE'], data.iloc[:, -4:].to_numpy()
    BATCH_SIZE = 32
    EMBEDDED_SIZE = 5
    GRU_HIDDEN_SIZE = 64
    NUM_OUTPUT_NODES = 4
    NUM_LAYERS = 1
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    INPUT_SIZE = 4100

    """
    Tokenize data and create vocabulary
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts=X)
    sequences = tokenizer.texts_to_sequences(texts=X)
    X = pad_sequences(sequences, maxlen=INPUT_SIZE)

    """## Split data"""
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=2023)

    tensor_X_train = torch.tensor(X_train)
    tensor_X_val = torch.tensor(X_val)
    tensor_X_test = torch.tensor(X_test)
    tensor_Y_train = torch.FloatTensor(Y_train)
    tensor_Y_val = torch.FloatTensor(Y_val)
    tensor_Y_test = torch.FloatTensor(Y_test)

    train_dataset = TensorDataset(tensor_X_train, tensor_Y_train)
    val_dataset = TensorDataset(tensor_X_val, tensor_Y_val)
    test_dataset = TensorDataset(tensor_X_test, tensor_Y_test)

    data_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    data_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    data_test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    """## Run"""
    SIZE_OF_VOCAB = len(tokenizer.word_index.keys())
    model = Escort(SIZE_OF_VOCAB, EMBEDDED_SIZE, GRU_HIDDEN_SIZE, NUM_LAYERS, NUM_OUTPUT_NODES)
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    train_accuracies, valid_accuracies, train_losses, valid_losses = train(EPOCHS, model, optimizer, criterion, (data_train_loader, data_val_loader))

    plot_graph(EPOCHS, train_losses, valid_losses, "Train/Validation Loss")
    plot_graph(EPOCHS, train_accuracies, valid_accuracies, "Train/Validation Accuracy")

    total_preds, total_labels, execution_time = predict(data_test_loader, model)

    print('Execution time: ', execution_time)
    save_classification(y_pred=np.array(total_preds), y_test=np.array(total_labels), labels=labels, out_dir='./report/escort.csv')

if __name__ == '__main__':
   run()