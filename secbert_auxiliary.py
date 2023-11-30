import torch
import os
import pandas as pd
from transformers import BertModel, BertTokenizerFast
from dscv.models.models import BaseModel
from dscv.datasets import FullAuxiliaryOpcode
from dscv.utils.util import calculate_score, get_misclassified_data
import time

from torch.utils.data import DataLoader

if torch.cuda.is_available():
 dev = "cuda"
else:
 dev = "cpu"
device = torch.device(dev)
print(device)

def freeze_k_layer(secBert, k=1):
  for param in secBert.encoder.layer[0:k].parameters():
    param.requires_grad = False

def train_steps(training_loader, model, loss_f, optimizer):
    print('Training...')
    training_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    train_acc = 0.
    train_f1 = 0.
    misclassify_train_data = {}

    model.train()

    for step, batch in enumerate(training_loader):
        # push the batch to gpu
        indices = batch['index'].numpy()
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        tfidf_features = batch['tfidf_features'].to(device)
        word2vec = batch['word2vec'].to(device)
        targets = batch['targets'].to(device)

        preds, max_indices = model(ids, attention_mask=mask, token_type_ids=token_type_ids, tfidf_features=tfidf_features,word2vec=word2vec)

        # calculate the loss for each branch
        losses = [loss_f(preds[i], targets[:, i]) for i in range(targets.shape[1])]
        average_loss = sum(losses) / targets.shape[1]
        training_loss += average_loss.item()

        label_ids = targets.to('cpu').numpy()
        max_indices = max_indices.detach().cpu().numpy()
        acc_score = calculate_score(label_ids, max_indices)
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

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []
    total_labels = []
    # iterate over batches
    for step, batch in enumerate(validating_loader):
        # push the batch to gpu
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        tfidf_features = batch['tfidf_features'].to(device)
        word2vec = batch['word2vec'].to(device)
        targets = batch['targets'].to(device)
        
        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds, max_indices = model(ids, attention_mask=mask, token_type_ids=token_type_ids, tfidf_features=tfidf_features,word2vec=word2vec)

            # compute the validation loss between actual and predicted values
            losses = [loss_f(preds[i], targets[:, i]) for i in range(targets.shape[1])]
            average_loss = sum(losses) / targets.shape[1]
            total_loss += average_loss.item()

            max_indices = max_indices.detach().cpu().numpy()
            total_preds += list(max_indices)
            total_labels += targets.tolist()
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(validating_loader)
    acc_score = calculate_score(total_labels, total_preds)

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
        tfidf_features = batch['tfidf_features'].to(device)
        word2vec = batch['word2vec'].to(device)
        targets = batch['targets'].to(device)

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds, max_indices = model(ids, attention_mask=mask, token_type_ids=token_type_ids, tfidf_features=tfidf_features,word2vec=word2vec)

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

if __name__ == '__main__':
    # Defining some key variables that will be used later on in the training
    max_length = 512
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    EPOCHS = 1
    LEARNING_RATE = 1e-04
    num_class = 4
    labels = ['Timestamp dependence', 'Outdated Solidity version', 'Frozen Ether', 'Delegatecall Injection']
    data_folder = os.getcwd()+'/data/'

    X_train = pd.read_csv(data_folder+'X_train.csv')
    X_test = pd.read_csv(data_folder+'X_test.csv')
    X_val = pd.read_csv(data_folder+'X_val.csv')

    y_train = pd.read_csv(data_folder+'y_train.csv').to_numpy()
    y_test = pd.read_csv(data_folder+'y_test.csv').to_numpy()
    y_val = pd.read_csv(data_folder+'y_val.csv').to_numpy()

    secBertTokenizer = BertTokenizerFast.from_pretrained("jackaduma/SecBERT", do_lower_case=True)
    secBertModel = BertModel.from_pretrained("jackaduma/SecBERT", num_labels = num_class)
    freeze_k_layer(secBertModel, k=6)
    secBertClassifierMultilabel = BaseModel(original_model=secBertModel, num_classes=num_class)
    training_set = FullAuxiliaryOpcode(X_train, y_train, secBertTokenizer, max_length)
    validating_set = FullAuxiliaryOpcode(X_val, y_val, secBertTokenizer, max_length)
    testing_set = FullAuxiliaryOpcode(X_test, y_test, secBertTokenizer, max_length)

    # Create generator for Dataset with BATCH_SIZE
    training_loader = DataLoader(training_set, batch_size=TRAIN_BATCH_SIZE)
    validating_loader = DataLoader(validating_set, batch_size=VALID_BATCH_SIZE)
    testing_loader = DataLoader(testing_set, batch_size=VALID_BATCH_SIZE)