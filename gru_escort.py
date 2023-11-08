import pandas as pd
import os
import collections
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.feature_extraction_utils import BagOfWord
from utils.process_text import Tokenizer, pad_sequences

import pickle

if torch.cuda.is_available():
 dev = "cuda:0"
else:
 dev = "cpu"
device = torch.device(dev)
device

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

"""# Extract and Read Data"""

data_folder = os.getcwd()+'/data-multilabel/'

class OpcodeData(Dataset):
    def __init__(self, X, y, tokenizer, max_len, tfidf_vectorizer):
        self.tokenizer = tokenizer
        self.X = X
        self.targets = y
        self.max_len = max_len
        self.tfidf_vectorizer = tfidf_vectorizer

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        text = str(self.X[index])

        tfidf = self.tfidf_vectorizer.transform([text])
        sequence = tokenizer.texts_to_sequences(texts=[text])
        ids = pad_sequences(sequence, maxlen=input_size)


        return {
            'ids': torch.tensor(ids.flatten(), dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float),
            'tfidf': torch.tensor(tfidf.flatten(), dtype=torch.float),
        }

"""# Feature extraction"""

# TF-IDF
class TfIdf:
  def __init__(self, save_model_dir):
    self.save_model_dir = save_model_dir
    try:
      self.tfidf_vectorizer = self.load_model(self.save_model_dir)
      print(f"Load model from {self.save_model_dir}")
    except:
      self.tfidf_vectorizer = TfidfVectorizer()
      print(f"Create new model for training")

  def train_ifidf(self, X_train):
    self.tfidf_vectorizer.fit_transform(raw_documents=X_train)
    self.save_model(self.tfidf_vectorizer)

  def transform(self, raw_documents):
    return self.tfidf_vectorizer.transform(raw_documents=raw_documents).toarray()

  def save_model(self, tfidf_vectorizer):
    try:
      with open(self.save_model_dir, 'wb') as file:
        pickle.dump(tfidf.tfidf_vectorizer, file)
      print(f'save successfully TfidfVectorizer with {self.save_model_dir}')

    except:
      print(f'can\'t save TfidfVectorizer with {self.save_model_dir}')

  def load_model(self, save_model_dir):
    with open(save_model_dir, 'rb') as file:
      tfidf_vectorizer = pickle.load(file)

    return tfidf_vectorizer

"""# GRU + TFIDF + ESCORT"""

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

"""## ESCORT"""

class Escort(nn.Module):
  def __init__(self, vocab_size, embedd_size, gru_hidden_size, n_layers, num_classes):
    super(Escort, self).__init__()
    self.word_embeddings = nn.Embedding(vocab_size, embedd_size)
    self.gru = nn.GRU(embedd_size, gru_hidden_size, num_layers=n_layers)
    self.branches = nn.ModuleList([Branch(gru_hidden_size, 128, 64, 0.2, 1) for _ in range(num_classes)])
    self.sigmoid = nn.Sigmoid()

  def forward(self, sequence, tfidf_inputs):
    embeds = self.word_embeddings(sequence)
    gru_out, _ = self.gru(embeds)
    last_hidden_state = gru_out[:, -1, :]
    pooler_input = last_hidden_state + tfidf_inputs
    output_branches = [branch(pooler_input) for branch in self.branches]
    output_branches = torch.cat(output_branches, dim=1)
    outputs = self.sigmoid(output_branches)
    return outputs

"""## Train model"""
"""### Train and Validation Steps"""

def calculate_score(y_true, preds):
    acc_score = accuracy_score(y_true, preds)

    return acc_score

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

X_train = pd.read_csv(data_folder+'X_train.csv')['BYTECODE'].sample(frac=0.1).to_numpy()
X_test = pd.read_csv(data_folder+'X_test.csv')['BYTECODE'].sample(frac=0.1).to_numpy()
X_val = pd.read_csv(data_folder+'X_val.csv')['BYTECODE'].sample(frac=0.1).to_numpy()

y_train = pd.read_csv(data_folder+'y_train.csv').sample(frac=0.1).to_numpy()
y_test = pd.read_csv(data_folder+'y_test.csv').sample(frac=0.1).to_numpy()
y_val = pd.read_csv(data_folder+'y_val.csv').sample(frac=0.1).to_numpy()

# save_idf_dir_file = os.path.join(os.getcwd() + '/trained/tfidf/', 'tfidf_vectorizer.pkl')
# tfidf = TfIdf(save_model_dir=save_idf_dir_file)
# tfidf.train_ifidf(X_train=X_train)

print("Feature Extraction - Bag Of Word")
bow = BagOfWord(X_train=X_train, X_test=X_test)
X_train_bow, X_test_bow = bow()

GRU_HIDDEN_SIZE = len(bow.vectorizer.vocabulary_)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts=X_train)
SIZE_OF_VOCAB = len(tokenizer.word_index.keys())

# train_dataset = OpcodeData(X_train, y_train, tokenizer, input_size, tfidf)
# val_dataset = OpcodeData(X_val, y_val, tokenizer, input_size, tfidf)
# test_dataset = OpcodeData(X_test, y_test, tokenizer, input_size, tfidf)
train_dataset = OpcodeData(X_train, y_train, tokenizer, input_size, bow)
val_dataset = OpcodeData(X_val, y_val, tokenizer, input_size, bow)
test_dataset = OpcodeData(X_test, y_test, tokenizer, input_size, bow)

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
