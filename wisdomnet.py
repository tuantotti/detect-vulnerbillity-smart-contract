import torch
from dscv.utils.util import get_misclassified_data, cal_wisdomnet_acc
from dscv.utils.save_report import wisdomnet_classification_report, wisdomnet_cfm
from dscv.models.models import WisdomNet
from dscv.datasets import OpcodeData
from dscv.models.models import BaseModel
from dscv.utils.util import plot_graph
from transformers import BertModel, BertTokenizerFast
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import numpy as np
import pandas as pd

if torch.cuda.is_available():
 dev = "cuda"
else:
 dev = "cpu"
device = torch.device(dev)

def train_steps(training_loader, model, loss_f, optimizer, score_type='', is_find_miss=True):
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
        targets = batch['targets'].to(device)

        preds, max_indices = model(ids, attention_mask=mask, token_type_ids=token_type_ids)

        # calculate the loss for each branch
        losses = [loss_f(preds[i], targets[:, i]) for i in range(targets.shape[1])]
        average_loss = sum(losses) / targets.shape[1]
        training_loss += average_loss.item()

        label_ids = targets.to('cpu').numpy()
        max_indices = max_indices.detach().cpu().numpy()
        acc_score = cal_wisdomnet_acc(label_ids, max_indices, score_type)
        train_acc += acc_score

        if is_find_miss:
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

def evaluate_steps(validating_loader, model, loss_f, score_type='', **kwargs):
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
    acc_score = cal_wisdomnet_acc(total_labels, total_preds, score_type)

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

def train(epochs, model, optimizer, criterion, dataloader, score_type='', save_dir='secbert-escort.pt', is_find_miss=True):
  data_train_loader, data_val_loader, data_test_loader = dataloader
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
    train_loss, train_acc, misclassify_train_steps_data = train_steps(data_train_loader, model, criterion, optimizer, score_type=score_type, is_find_miss=is_find_miss)
    valid_loss, valid_acc = evaluate_steps(data_val_loader, model, criterion, score_type=score_type)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), save_dir)
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_acc)
    valid_accuracies.append(valid_acc)
    misclassify_train_data.update(misclassify_train_steps_data)

    elapsed_time = time.time() - start_time

    print('\t loss={:.4f} \t accuracy={:.4f} \t val_loss={:.4f}  \t val_acc={:.4f}  \t time={:.2f}s'.format(train_loss, train_acc, valid_loss, valid_acc, elapsed_time))

    labels = ['Timestamp dependence', 'Outdated Solidity version', 'Frozen Ether', 'Delegatecall Injection']
    total_labels, total_preds = predict(data_test_loader, model)
    print(wisdomnet_classification_report(np.array(total_labels), np.array(total_preds), labels))

  return train_accuracies, valid_accuracies, train_losses, valid_losses, misclassify_train_data

def get_mis_data(misclassified_df, frac=0.01):
  # df = pd.DataFrame(columns=misclassified_df.columns)
  # labels = misclassified_df.columns[-4:]
  # reject_label = 2

  # def find_min_reject(misclassified_df):
  #   num_reject_counts = misclassified_df.apply(lambda x: (x == 2).sum())
  #   min_num_reject = num_reject_counts.min()
  #   min_index = num_reject_counts.argmin()
  #   min_label = num_reject_counts.index[min_index]

  #   return min_label, min_num_reject

  # min_label, min_num_reject = find_min_reject(misclassified_df.iloc[:, -4:])
  # threshold = int(min_num_reject * frac)

  # for label in labels:
  #   df_filtered = misclassified_df.loc[misclassified_df[label] == reject_label, :]
  #   df = pd.concat([df, df_filtered[:threshold]])

  df = misclassified_df.sample(frac=frac, random_state=2023)
  return df

def freeze_secbert_layer(secBert):
  for param in secBert.embeddings.parameters():
    param.requires_grad = False
  for param in secBert.encoder.parameters():
    param.requires_grad = False
  for param in secBert.pooler.parameters():
    param.requires_grad = False
    
# Freeze all branch except the last layer
def freeze_branch(branches):
  for branch in branches:
    for param in branch.parameters():
      param.dense1.weight.requires_grad = False
      param.dense1.bias.requires_grad = False
      param.dense2.weight.requires_grad = False
      param.dense2.bias.requires_grad = False
      
def freeze_all_layer(wisdomnet):
  freeze_secbert_layer(wisdomnet.bert)
  freeze_branch(wisdomnet.new_branches)
  
def misclassified_analysis(misclassified_df):
  values = []
  keys = [0 ,1, 2]
  labels = misclassified_df.columns[-4:]
  for i in labels:
    value_counts = misclassified_df[i].value_counts().to_dict()
    value = [0, 0, 0]
    for key in keys:
      if key in value_counts.keys():
        value[key] = value_counts[key]
      else:
        value[key] = 0

    values.append(value)

  values = np.array(values).T
  values.shape[0]

  # Plot
  fig, ax = plt.subplots(figsize=(16, 9))
  name = ['No', 'Yes', 'Reject']
  # Stacked bar chart
  for i in range(values.shape[0]):
    ax.bar(labels, values[i], bottom = np.sum(values[:i], axis = 0), width = 0.5, label = str(name[i]))

  for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2 + bar.get_y(),
            round(bar.get_height()), ha = 'center',
            color = 'black', weight = 'bold', size = 10)

  plt.legend()
  ax.set_ylabel('Total')
  ax.set_xlabel('Name of vulnerbilities')
  plt.show()
  

def train_wisdomnet(frac_mis=0.01):
    # Defining some key variables that will be used later on in the training
    max_length = 512
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-04
    num_class = 4
    sebert_escort_dir = '/home/tzt5387/Desktop/secbert-escort.pt'
    mis_classified_data = '/home/tzt5387/Desktop/Untitled Folder/misclassified-outdate.csv'
    data_folder = '/home/tzt5387/Desktop/Untitled Folder/'
    save_result_folder = ''

    secBertTokenizer = BertTokenizerFast.from_pretrained("jackaduma/SecBERT", do_lower_case=True)
    secBertModel = BertModel.from_pretrained("jackaduma/SecBERT")
    
    print('Load trained model')
    secBertClassifierMultilabel = BaseModel(original_model=secBertModel, num_classes=num_class)
    secBertClassifierMultilabel.load_state_dict(torch.load(sebert_escort_dir))
    # freeze_all_layer(secBertClassifierMultilabel)

    wisdomnet = WisdomNet(secBertClassifierMultilabel)
    freeze_all_layer(wisdomnet)
    wisdomnet.to(device)

    # data = pd.read_csv(data_folder + '/Data_Cleansing_a.csv')
    print('Load misclassified outdated data')
    misclassified_df = pd.read_csv(mis_classified_data)
    mis_df = get_mis_data(misclassified_df, frac=frac_mis)
    X_m_index = mis_df['X_train_index'].to_list()
    X_train = pd.read_csv(data_folder+'X_train.csv')

    X_m, y_m = X_train.iloc[X_m_index].to_numpy(), np.array(mis_df.iloc[:, -4:].to_numpy(), dtype='int64')
    X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_m, y_m, test_size=0.2, random_state=2023)
    X_m_train, X_m_val, y_m_train, y_m_val = train_test_split(X_m_train, y_m_train, test_size=0.2, random_state=2023)

    training_m_set = OpcodeData(X_m_train, y_m_train, secBertTokenizer, max_length)
    validating_m_set = OpcodeData(X_m_val, y_m_val, secBertTokenizer, max_length)
    testing_m_set = OpcodeData(X_m_test, y_m_test, secBertTokenizer, max_length)

    # Create generator for Dataset with BATCH_SIZE
    training_m_loader = DataLoader(training_m_set, batch_size=TRAIN_BATCH_SIZE)
    validating_m_loader = DataLoader(validating_m_set, batch_size=VALID_BATCH_SIZE)
    testing_m_loader = DataLoader(testing_m_set, batch_size=VALID_BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = wisdomnet.parameters(), lr=LEARNING_RATE)

    print('Train Wisdom Net Model')
    train_m_accuracies, valid_m_accuracies, train_m_losses, valid_m_losses, _ = train(EPOCHS, wisdomnet, optimizer, criterion, (training_m_loader, validating_m_loader, testing_m_loader), score_type='multioutput', save_dir='wisdomnet-escort'+str(frac_mis*100)+'-percent.pt', is_find_miss=False)

    """
    Plot the result of training process
    """
    plot_graph(EPOCHS, train_m_losses, valid_m_losses, "Train/Validation Loss")
    plot_graph(EPOCHS, train_m_accuracies, valid_m_accuracies, "Train/Validation Accuracy")

    """
    Evaluate model on test set and save the result
    """
    print('Test Wisdom Net Model')
    X_test = pd.read_csv(data_folder+'X_test.csv').to_numpy()
    y_test = pd.read_csv(data_folder+'y_test.csv').to_numpy()

    testing_set = OpcodeData(X_test, y_test, secBertTokenizer, max_length)
    testing_loader = DataLoader(testing_set, batch_size=VALID_BATCH_SIZE)

    labels = ['Timestamp dependence', 'Outdated Solidity version', 'Frozen Ether', 'Delegatecall Injection']
    start = time.time()
    total_labels, total_preds = predict(testing_loader, wisdomnet)
    end = time.time()

    execution_time = (end - start) / len(total_labels)
    print('Execution time: ', execution_time)
    df_labels = pd.DataFrame(total_labels, columns=labels)
    df_preds = pd.DataFrame(total_preds, columns=labels)

    print(wisdomnet_cfm(df_labels.to_numpy(), df_preds.to_numpy()))
    mycr = wisdomnet_classification_report(df_labels.to_numpy(), df_preds.to_numpy(), labels)
    mycr.to_csv(save_result_folder+str(frac_mis*100)+'-percent.csv')
    print(mycr)

if __name__ == '__main__':
    train_wisdomnet(frac_mis=0.01)