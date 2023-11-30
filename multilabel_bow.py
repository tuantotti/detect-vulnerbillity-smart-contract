import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from dscv.models.ml_models import MultilabelModel
from dscv.utils.feature_extraction_utils import BagOfWord
from dscv.utils.save_report import save_classification
"""
Read and preprocess data
"""
print("Read and preprocess data")
data_folder = os.getcwd() + '/data/'
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

"""## Feature Extraction
### BOW
"""
print("Feature Extraction - Bag Of Word")
bow = BagOfWord(X_train=X_train, X_test=X_test)
X_train_bow, X_test_bow = bow()

"""# Multilabel machine learning
# ### Label Powerset
# """
print("Multilabel machine learning")
print("Label Powerset")
lbl_powerset = MultilabelModel(X_train=X_train_bow, y_train=Y_train, X_test=X_test_bow, 
                               method='LabelPowerset', num_classes=num_classes)
y_pred_lbp = lbl_powerset()

save_classification(y_test=Y_test, y_pred=y_pred_lbp.toarray(), out_dir='./report/Label_Powerset_BOW.csv', labels=labels)

"""### Binary relevence"""
print("Binary relevence")
# with a gaussian naive bayes base classifier
bin_relevence = MultilabelModel(X_train=X_train_bow, y_train=Y_train, X_test=X_test_bow, 
                               method='BinaryRelevance', num_classes=num_classes)
y_pred_binre = bin_relevence()
save_classification(y_test=Y_test, y_pred=y_pred_binre.toarray(), out_dir='./report/Binary_Relevence_BOW.csv', labels=labels)

"""### Adapted Algorithm"""
print("Adapted Algorithm")
adapt_al = MultilabelModel(X_train=X_train_bow, y_train=Y_train, X_test=X_test_bow, 
                               method='MLkNN', num_classes=num_classes)
y_pred_adapt = adapt_al()
save_classification(y_test=Y_test, y_pred=y_pred_adapt.toarray().astype(int), out_dir='./report/Adapted_Algorithm_BOW.csv', labels=labels)

"""### Classifier Chains"""
print("Classifier Chains")
classifier_chain = MultilabelModel(X_train=X_train_bow, y_train=Y_train, X_test=X_test_bow, 
                               method='ClassifierChain', num_classes=num_classes)
Y_pred_chains = classifier_chain()
save_classification(y_test=Y_test, y_pred=Y_pred_chains.astype(int), out_dir='./report/Classifier_Chains_BOW.csv', labels=labels)