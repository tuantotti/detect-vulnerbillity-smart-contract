import pandas as pd
import os
import numpy as np

from dscv.models.ml_models import MultilabelModel
from dscv.utils.feature_extraction_utils import TfIdf
from dscv.utils.save_report import save_classification
from dscv.utils.process_text import Tokenizer, pad_sequences
from dscv.utils.feature_extraction_utils import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

print("Read data")
X_train = pd.read_csv('./data-multilabel/X_train.csv')
X_test = pd.read_csv('./data-multilabel/X_test.csv')
X_valid = pd.read_csv('./data-multilabel/X_val.csv')
y_train = pd.read_csv('./data-multilabel/y_train.csv')
y_test = pd.read_csv('./data-multilabel/y_test.csv')
y_valid = pd.read_csv('./data-multilabel/y_val.csv')

X_train = pd.concat([X_train, X_valid])
y_train = pd.concat([y_train, y_valid])
labels = y_train.columns.to_list()

print(X_train.shape)
print(y_train.shape)

print("Feature Extraction - TFIDF")
tfidf = TfIdf(X_train['BYTECODE'].copy(deep=False), X_test['BYTECODE'].copy(deep=False))
X_train_idf, X_test_idf = tfidf()
print(X_train_idf.shape)

print("Feature Extraction - W2v")
max_length = X_train_idf.shape[1]
tokenizer = Tokenizer(lower=False)

# Create vocabulary
tokenizer.fit_on_texts(X_train['BYTECODE'].copy(deep=False))
print(type(X_test['BYTECODE']))
# Transforms each text in texts to a sequence of integers
sequences_train = tokenizer.texts_to_sequences(X_train['BYTECODE'].copy(deep=False))
sequences_test = tokenizer.texts_to_sequences(X_test['BYTECODE'].copy(deep=False))
# Pads sequences to the same length
X_tokenized_train = pad_sequences(sequences_train, maxlen=max_length)
X_tokenized_test = pad_sequences(sequences_test, maxlen=max_length)
print(X_tokenized_train.shape)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
word2vec = Word2Vec(word_index)
# print('Train embedding')
# word2vec.train_vocab(X=X_train, embedding_dim=32)
embedding_matrix = word2vec()
print(embedding_matrix.shape)

# create mean for machine learning
mean_embedding = embedding_matrix.mean(axis=1)
X_mean_embedding_train = X_tokenized_train.copy().astype('float32')
for i, x in enumerate(X_tokenized_train):
    for j, value in enumerate(x):
        X_mean_embedding_train[i, j] = mean_embedding[value]

X_mean_embedding_test = X_tokenized_test.copy().astype('float32')
for i, x in enumerate(X_tokenized_test):
    for j, value in enumerate(x):
        X_mean_embedding_test[i, j] = mean_embedding[value]

scaler = MinMaxScaler()
X_train_w2v = scaler.fit_transform(X_mean_embedding_train)
X_test_w2v = scaler.fit_transform(X_mean_embedding_test)
print(X_train_w2v.shape)
print(f'{X_train_w2v.shape == X_train_idf.shape}')

X_train = X_train_w2v + X_train_idf
X_test = X_test_w2v + X_test_idf
"""### Adapted Algorithm"""
print("Adapted Algorithm")
num_classes = 4
adapt_al = MultilabelModel(X_train=X_train, y_train=y_train.to_numpy(), X_test=X_test, 
                               method='MLkNN', num_classes=num_classes)

y_pred_adapt = adapt_al()
save_classification(y_test=y_test.to_numpy(), y_pred=y_pred_adapt, out_dir='./report/Adapted_Algorithm_W2V_TFIDF.csv', labels=labels)

