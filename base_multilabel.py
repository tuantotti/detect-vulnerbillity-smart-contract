from typing import Any
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from skmultilearn.adapt import MLkNN
import numpy as np
import time


class MultilabelModel:
    def __init__(self, X_train, y_train, X_test, method='LabelPowerset', num_classes=4) -> None:
        self.method = method
        self.num_classes = num_classes
        self.X_train, self.y_train, self.X_test= X_train, y_train, X_test

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        y_pred = []
        if self.method == 'LabelPowerset':
            classifier = LabelPowerset(LogisticRegression())
            classifier.fit(self.X_train, self.y_train)

            # Predict and calculate execution time
            start_time = time.time()
            y_pred = classifier.predict(self.X_test)
            execution_time = (time.time() - start_time) / len(self.X_test)
            print('Execution time: ', execution_time)

        elif self.method == 'BinaryRelevance':
            classifier = BinaryRelevance(GaussianNB())
            classifier.fit(self.X_train, self.y_train)

            # Predict and calculate execution time
            start_time = time.time()
            y_pred = classifier.predict(self.X_test)
            execution_time = (time.time() - start_time) / len(self.X_test)
            print('Execution time: ', execution_time)

        elif self.method == 'ClassifierChain':
            base_lr = LogisticRegression()

            chains = [ClassifierChain(base_lr, order="random", random_state=i) for i in range(self.num_classes)]
            for chain in chains:
                chain.fit(self.X_train, self.y_train)

            # Predict and calculate execution time
            start_time = time.time()
            Y_pred_chains = np.array([chain.predict(self.X_test) for chain in chains])
            execution_time = (time.time() - start_time) / len(self.X_test)
            print('Execution time: ', execution_time)

            y_pred = Y_pred_chains.mean(axis=0)

        elif self.method == 'MLkNN':
            classifier = MLkNN(k=self.num_classes)
            classifier.fit(self.X_train, self.y_train)

            # Predict and calculate execution time
            start_time = time.time()
            y_pred = classifier.predict(self.X_test)
            execution_time = (time.time() - start_time) / len(self.X_test)
            print('Execution time: ', execution_time)

        else:
            print('No method chosen')

        return y_pred
