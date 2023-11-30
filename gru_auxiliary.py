import pandas as pd
import numpy as np
import os
import torch.nn as nn
from dscv.utils.util import calculate_score
from dscv.utils.util import get_misclassified_data_v2
from dscv.utils.feature_extraction_utils import Word2Vec, TfIdf
from dscv.utils.process_text import Tokenizer, pad_sequences
from dscv.utils.util import plot_graph
from dscv.datasets import EscortOpcodeData
from torch.utils.data import DataLoader
from dscv.models.models import Escort
from dscv.utils.save_report import save_classification

import torch

from sklearn.metrics import *
import time

if torch.cuda.is_available():
 dev = "cuda:0"
else:
 dev = "cpu"
device = torch.device(dev)

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
        indices = batch['index']
        tfidf_features = batch['tfidf_features'].to(device)
        ids = batch['ids'].to(device)
        word2vec = batch['word2vec'].to(device)
        targets = batch['targets'].to(device)

        preds = model(ids, tfidf_features=tfidf_features,word2vec=word2vec)
        
        loss = loss_f(preds, targets)
        training_loss += loss.item()

        preds = preds.detach().cpu().numpy()
        preds = np.where(preds>=0.5, 1, 0)
        targets = targets.to('cpu').numpy()

        acc_score = calculate_score(targets, preds)
        
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
    print("Evaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []
    total_labels = []
    total_opcodes = []
    # iterate over batches
    for step, batch in enumerate(validating_loader):
        # push the batch to gpu
        opcode = batch['opcode']
        tfidf_features = batch['tfidf_features'].to(device)
        ids = batch['ids'].to(device)
        word2vec = batch['word2vec'].to(device)
        targets = batch['targets'].to(device)
        
        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(ids, tfidf_features=tfidf_features,word2vec=word2vec)

            # compute the validation loss between actual and predicted values
            loss = loss_f(preds, targets)
            total_loss += loss.item()

            preds = preds.detach().cpu().numpy()
            preds = np.where(preds>=0.5, 1, 0)
            total_preds += list(preds)
            total_opcodes += list(opcode)
            total_labels += targets.tolist()
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(validating_loader)
    acc_score = accuracy_score(total_labels, total_preds)
    misclassified_data = get_misclassified_data_v2(np.array(total_labels), np.array(total_preds), total_opcodes)

    return avg_loss, acc_score, misclassified_data

def train(epochs, model, optimizer, criterion, dataloader, save_model_dir):
  data_train_loader, data_val_loader = dataloader
  # set initial loss to infinite
  best_valid_loss = float('inf')
  train_losses = []
  valid_losses = []
  train_accuracies = []
  valid_accuracies = []
  misclassify_train_data = []
  total_time = 0.0

  for epoch in range(epochs):
    print('Epoch {}/{} '.format(epoch + 1, epochs))
    start_time = time.time()
    train_loss, train_acc = train_steps(data_train_loader, model, criterion, optimizer)
    valid_loss, valid_acc, misclassified_val_steps_data = evaluate_steps(data_val_loader, model, criterion)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), save_model_dir)
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_acc)
    valid_accuracies.append(valid_acc)
    misclassify_train_data.extend(misclassified_val_steps_data)

    elapsed_time = time.time() - start_time
    total_time += elapsed_time

    print('\t loss={:.4f} \t accuracy={:.4f} \t val_loss={:.4f}  \t val_acc={:.4f}  \t time={:.2f}s'.format(train_loss, train_acc, valid_loss, valid_acc, elapsed_time))
  print(f'Total time: {total_time}')
  return train_accuracies, valid_accuracies, train_losses, valid_losses, misclassify_train_data


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
        tfidf_features = batch['tfidf_features'].to(device)
        ids = batch['ids'].to(device)
        word2vec = batch['word2vec'].to(device)
        targets = batch['targets'].to(device)
        
        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(ids, tfidf_features=tfidf_features,word2vec=word2vec)
            preds = preds.detach().cpu().numpy()
            preds = np.where(preds>=0.5, 1, 0)
            total_preds += list(preds)
            total_labels += targets.tolist()

    return total_labels, total_preds

if __name__ == "__main__":
    data_folder= os.getcwd()+'/data/'
    out_folder ='/saved_model/'
    report_folder = os.getcwd()+'/report/'
    # Define constant
    input_size = 4100
    epochs = 20
    SIZE_OF_VOCAB = 512
    EMBEDDED_SIZE = 5
    GRU_HIDDEN_SIZE = 256
    NUM_OUTPUT_NODES = 4
    NUM_LAYERS = 1
    DROPOUT = 0.2
    RNN_TYPE = 'GRU'
    BIDIRECTIONAL = False
    USE_MULTIBRANCHES = True
    LEARNING_RATE=1e-3
    labels = ['Timestamp dependence', 'Outdated Solidity version', 'Frozen Ether', 'Delegatecall Injection']
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    NUM_AUXILIARY=2 # this field use to define the number of additional feature. Max this number = 2 (w2v, tfidf)
                    # NUM_AUXILIARY = 1 when you use w2v or tfidf as a additional vector
                    # NUM_AUXILIARY = 2 when you use w2v and tfidf as a additional vector
    AUXILIARY_FEATURE_LENGTH=256 # this field use to define the number of additional vector 
                                # in the output vector of model
                                # For example: We use GRU and output is (32, 256) 
                                # and AUXILIARY_FEATURE_LENGTH = 256, NUM_AUXILIARY = 2 (w2v and tfidf)
                                # so the auxiliary vector is (32, 256+256*2) = (32, 768)
    save_model_dir = out_folder + 'escort-tfidf-w2v.pt'
    report_dir = report_folder + 'escort-tfidf-w2v.csv'
    tokenizer_dir = out_folder + 'tokenizer.pickle'
    w2v_dir = out_folder + 'fasttext_w2v.pickle'
    tfidf_dir = out_folder + 'tfidf.pickle'
    
    # Read data
    X_train = pd.read_csv(data_folder+'X_train.csv')
    X_test = pd.read_csv(data_folder+'X_test.csv')
    X_val = pd.read_csv(data_folder+'X_val.csv')

    y_train = pd.read_csv(data_folder+'y_train.csv').to_numpy()
    y_test = pd.read_csv(data_folder+'y_test.csv').to_numpy()
    y_val = pd.read_csv(data_folder+'y_val.csv').to_numpy()
    
    # load feature extraction if you have not trained yet --> train 
    tokenizer = Tokenizer.load_model(tokenizer_dir)
    word2vec_model = Word2Vec.load_model(w2v_dir)
    tfidf_model = TfIdf.load_model(tfidf_dir)
    
    train_dataset = EscortOpcodeData(X=X_train["BYTECODE"], y=y_train, tokenizer=tokenizer, tfidf=tfidf_model, w2v_model=word2vec_model, max_len=input_size)
    test_dataset = EscortOpcodeData(X=X_test["BYTECODE"], y=y_test, tokenizer=tokenizer, tfidf=tfidf_model, w2v_model=word2vec_model, max_len=input_size)
    val_dataset = EscortOpcodeData(X=X_val["BYTECODE"], y=y_val, tokenizer=tokenizer, tfidf=tfidf_model, w2v_model=word2vec_model, max_len=input_size)
    
    # Create generator for Dataset with BATCH_SIZE
    training_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE)
    validating_loader = DataLoader(val_dataset, batch_size=VALID_BATCH_SIZE)
    testing_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE)
    
    # Define model
    model = Escort(SIZE_OF_VOCAB, EMBEDDED_SIZE, GRU_HIDDEN_SIZE, NUM_LAYERS, NUM_OUTPUT_NODES, RNN_TYPE, BIDIRECTIONAL, USE_MULTIBRANCHES, NUM_AUXILIARY, AUXILIARY_FEATURE_LENGTH)
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    # Train model
    train_accuracies, valid_accuracies, train_losses, valid_losses, misclassify_train_data = train(epochs, model, optimizer, criterion, (training_loader, validating_loader), save_model_dir)
    
    # Plot graph
    plot_graph(epochs, train_losses, valid_losses, "Train/Validation Loss")
    plot_graph(epochs, train_accuracies, valid_accuracies, "Train/Validation Accuracy")
    
    # Test model
    total_preds, total_labels, execution_time = predict(testing_loader, model, criterion)
    print(f"Execution time: ${execution_time}")
    save_classification(y_pred=np.array(total_preds), y_test=np.array(total_labels), labels=labels, out_dir=report_dir)
    
    