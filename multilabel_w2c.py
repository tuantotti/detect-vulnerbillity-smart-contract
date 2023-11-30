import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from dscv.models.ml_models import MultilabelModel
from dscv.utils.save_report import save_classification
from dscv.utils.feature_extraction_utils import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from dscv.utils.process_text import Tokenizer, pad_sequences


"""
Read and preprocess data
"""
print("Read and preprocess data")
data_folder = os.getcwd() + '/data/'
data = pd.read_csv(data_folder + 'Data_Cleansing.csv')
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
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=2023)


"""## Feature Extraction
### Word2Vec
"""
print("Feature Extraction - Word2Vec")
max_length = 5500
tokenizer = Tokenizer(lower=False)

# Create vocabulary
tokenizer.fit_on_texts(X_train)

# Transforms each text in texts to a sequence of integers
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

# Pads sequences to the same length
X_tokenized_train = pad_sequences(sequences_train, maxlen=max_length)
X_tokenized_test = pad_sequences(sequences_test, maxlen=max_length)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
word2vec = Word2Vec(word_index)

"""
#### Train embedding
"""
print('Train embedding')
word2vec.train_vocab(X=X_train, embedding_dim=32)
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
X_train = scaler.fit_transform(X_mean_embedding_train)
X_test = scaler.fit_transform(X_mean_embedding_test)

"""# Multilabel machine learning
### Label Powerset
"""
print("Multilabel machine learning")
print("Label Powerset")
lbl_powerset = MultilabelModel(X_train=X_train, y_train=Y_train, X_test=X_test, 
                               method='LabelPowerset', num_classes=num_classes)
y_pred_lbp = lbl_powerset()

save_classification(y_test=Y_test, y_pred=y_pred_lbp.toarray(), out_dir='./report/Label_Powerset_W2V.csv', labels=labels)

"""### Binary relevence"""
print("Binary relevence")
# with a gaussian naive bayes base classifier
bin_relevence = MultilabelModel(X_train=X_train, y_train=Y_train, X_test=X_test, 
                               method='BinaryRelevance', num_classes=num_classes)
y_pred_binre = bin_relevence()
save_classification(y_test=Y_test, y_pred=y_pred_binre.toarray(), out_dir='./report/Binary_Relevence_W2V.csv', labels=labels)

"""### Classifier Chains"""
print("Classifier Chains")
classifier_chain = MultilabelModel(X_train=X_train, y_train=Y_train, X_test=X_test, 
                               method='ClassifierChain', num_classes=num_classes)
Y_pred_chains = classifier_chain()
save_classification(y_test=Y_test, y_pred=Y_pred_chains.astype(int), out_dir='./report/Classifier_Chains_W2V.csv', labels=labels)

"""### Adapted Algorithm"""
print("Adapted Algorithm")
adapt_al = MultilabelModel(X_train=X_train, y_train=Y_train, X_test=X_test, 
                               method='MLkNN', num_classes=num_classes)
y_pred_adapt = adapt_al()
save_classification(y_test=Y_test, y_pred=y_pred_adapt.toarray().astype(int), out_dir='./report/Adapted_Algorithm_W2V.csv', labels=labels)
