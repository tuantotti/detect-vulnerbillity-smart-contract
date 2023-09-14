import os
import pandas as pd
from sklearn.model_selection import train_test_split


data_folder = os.getcwd() + '/data-multilabel/'
data = pd.read_csv(data_folder + '/Data_Cleansing.csv')
selected_columns = ['BYTECODE', 'Timestamp dependence', 'Outdated Solidity version', 'Frozen Ether', 'Delegatecall Injection']
data = data.loc[:, selected_columns]

X, y = data['BYTECODE'], data.iloc[:, -4:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=2023)

X_train.to_csv(data_folder + 'X_train.csv')
X_test.to_csv(data_folder + 'X_test.csv')
X_val.to_csv(data_folder + 'X_val.csv')

y_train.to_csv(data_folder + 'y_train.csv')
y_test.to_csv(data_folder + 'y_test.csv')
y_val.to_csv(data_folder + 'y_val.csv')