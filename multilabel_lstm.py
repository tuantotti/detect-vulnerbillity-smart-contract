import numpy as np
import time
import os
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from utils.feature_extraction_utils import TfIdf, BagOfWord, Word2Vec
from sklearn.model_selection import train_test_split
from utils.save_report import save_classification


from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

import matplotlib.pyplot as plt

"""
Check GPU
"""
if torch.cuda.is_available():
 dev = "cuda:0"
else:
 dev = "cpu"
device = torch.device(dev)
print(device)

"""
Create mmultilabel model using LSTM
"""
class LSTMMultilabel(nn.Module):
  def __init__(self, vocab_size, hidden_dim, output_dim, n_layers, dropout, weight, use_embedding=False):
    super(LSTMMultilabel, self).__init__()

    self.use_embedding = use_embedding
    self.weight = weight
    print(dropout)

    if self.use_embedding:
      self.word_embeddings = nn.Embedding.from_pretrained(self.weight)
      self.word_embeddings.weight.requires_grad = False
      self.lstm = nn.LSTM(weight.shape[1], hidden_dim, num_layers=n_layers)
    else:
      self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers=n_layers)

    self.dense = nn.Linear(hidden_dim, output_dim)
    self.sigmoid = nn.Sigmoid()

  def forward(self, sequence):
    if self.use_embedding == False:
      lstm_out, _ = self.lstm(sequence)
      dense_out = self.dense(lstm_out)
    else:
      embeds = self.word_embeddings(sequence)
      lstm_out, _ = self.lstm(embeds)
      dense_out = self.dense(lstm_out[:, -1, :])

    outputs = self.sigmoid(dense_out)

    return outputs
  
"""
Traning and validation steps
"""
def calculate_score(y_true, preds):
    F1_score = f1_score(y_true, preds, average='macro')
    acc_score = accuracy_score(y_true, preds)

    return acc_score, F1_score

def train_steps(training_loader, model, loss_f, optimizer):
    training_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    train_acc = 0.
    train_f1 = 0.

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

        acc_score, F1_score = calculate_score(labels, preds)
        train_acc += acc_score
        train_f1 += F1_score
        nb_tr_steps += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # When using GPU
        optimizer.step()

    epoch_loss = training_loss / nb_tr_steps
    epoch_acc = train_acc / nb_tr_steps
    epoch_f1 = train_f1 / nb_tr_steps
    return epoch_loss, epoch_acc, epoch_f1

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
    acc_score, F1_score = calculate_score(total_labels, total_preds)

    return avg_loss, acc_score, F1_score

"""
Training loop
"""
def train(epochs, model, optimizer, criterion, dataloader):
  data_train_loader, data_val_loader = dataloader
  # empty lists to store training and validation loss of each epoch
  # set initial loss to infinite
  best_valid_loss = float('inf')
  train_losses = []
  valid_losses = []
  train_accuracies = []
  valid_accuracies = []

  if os.path.isdir('./trained'):
     os.mkdir('./trained')

  for epoch in range(epochs):
    start_time = time.time()
    train_loss, train_acc, _ = train_steps(data_train_loader, model, criterion, optimizer)
    
    valid_loss, valid_acc, _ = evaluate_steps(data_val_loader, model, criterion)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './trained/multilabel-lstm.pt')
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

"""
Predict 
"""
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


def run(feature_extraction_method='tfidf'):
  """
  Define constant
  """
  epochs = 10
  num_classes = 4
  NUM_HIDDEN_NODES = 128
  NUM_OUTPUT_NODES = num_classes
  NUM_LAYERS = 2
  BIDIRECTION = False
  DROPOUT = 0.2
  BATCH_SIZE = 128

  """
  Read and preprocess data
  """
  print("Read and preprocess data")
  data_folder = os.getcwd() + '/data-multilabel/'
  data = pd.read_csv(data_folder + '/Data_Cleansing.csv')
  data = data.drop(['Unnamed: 0', 'index', 'ADDRESS', 'LABEL_FORMAT'], axis=1)
  num_classes = 4
  selected_columns = ['BYTECODE', 'Timestamp dependence', 'Outdated Solidity version', 'Frozen Ether', 'Delegatecall Injection']
  data = data.loc[:, selected_columns]
  X, y = data['BYTECODE'], data.iloc[:, -num_classes:].to_numpy()
  print('y: ', y.shape)
  labels = data.iloc[:, -num_classes:].keys().tolist()
  values = np.sum(y, axis=0)

  print(dict(zip(labels, values)))

  """## Split data"""
  X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  """
  Feature Extraction
  """
  if feature_extraction_method == 'TFIDF':
    print("Feature Extraction - TFIDF")
    tfidf = TfIdf(X_train, X_test)
    X_train, X_test = tfidf()
    SIZE_OF_VOCAB = X_train.shape[1]

  elif feature_extraction_method == 'BOW':
    print("Feature Extraction - Bag Of Word")
    bow = BagOfWord(X_train=X_train, X_test=X_test)
    X_train, X_test = bow()
    SIZE_OF_VOCAB  = X_train.shape[1]
  
  elif feature_extraction_method == 'W2V':
    print("Feature Extraction - W2V")
    max_length = 5500
    tokenizer = Tokenizer(lower=False)

    # Create vocabulary
    tokenizer.fit_on_texts(X_train)

    # Transforms each text in texts to a sequence of integers
    sequences_train = tokenizer.texts_to_sequences(X_train)
    sequences_test = tokenizer.texts_to_sequences(X_test)

    # Pads sequences to the same length
    word_index = tokenizer.word_index
    SIZE_OF_VOCAB = len(word_index) + 1
    word2vec = Word2Vec(word_index)
    word2vec.train_vocab(X=X_train, embedding_dim=32)
    embedding_matrix = word2vec()

    X_train = pad_sequences(sequences_train, maxlen=max_length)
    X_test = pad_sequences(sequences_test, maxlen=max_length)

  X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=2023)

  """
  Prepare data
  """
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

  """
  Create model
  """
  if feature_extraction_method == 'W2V':
    model = LSTMMultilabel(vocab_size=SIZE_OF_VOCAB, hidden_dim=NUM_HIDDEN_NODES, 
                           output_dim=NUM_OUTPUT_NODES, n_layers=NUM_LAYERS, dropout=DROPOUT, 
                           weight=torch.Tensor(embedding_matrix), use_embedding=True) 
  else:
    model = LSTMMultilabel(SIZE_OF_VOCAB, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, NUM_LAYERS, BIDIRECTION, DROPOUT)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  criterion = nn.BCELoss()
  """
  Train model
  """
  train_accuracies, valid_accuracies, train_losses, valid_losses = train(epochs, model, optimizer, criterion, (data_train_loader, data_val_loader))

  """
  Plot the result of training process
  """
  plot_graph(epochs, train_losses, valid_losses, "Train/Validation Loss")
  plot_graph(epochs, train_accuracies, valid_accuracies, "Train/Validation Accuracy")

  """
  Evaluate model on test set and save the result
  """
  total_preds, total_labels, execution_time = predict(data_test_loader, model)
  print('Execution time: ', execution_time)
  save_classification(y_test=np.array(total_labels), y_pred=np.array(total_preds), labels=labels, out_dir='./report/LSTM_'+feature_extraction_method+'.csv')

"""
Run 
"""
if __name__ == '__main__':
  run(feature_extraction_method='TFIDF')  
  run(feature_extraction_method='BOW')  
  run(feature_extraction_method='W2V')  


