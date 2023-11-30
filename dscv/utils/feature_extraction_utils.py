import numpy as np
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
from typing import List

# TF-IDF
class TfIdf:
    def __init__(self, save_model_dir, is_train=True):
        self.save_model_dir = save_model_dir
        try:
            if is_train:
                self.tfidf_vectorizer = TfidfVectorizer()
                print(f"Create new model for training")
            else:
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

    @staticmethod
    def save_model(self):
        try:
            with open(self.save_model_dir, 'wb') as file:
                pickle.dump(self.tfidf_vectorizer, file)
            print(f'save successfully TfidfVectorizer with {self.save_model_dir}')

        except:
            print(f'can\'t save TfidfVectorizer with {self.save_model_dir}')

    def load_model(self, save_model_dir):
        with open(save_model_dir, 'rb') as file:
            tfidf_vectorizer = pickle.load(file)

        return tfidf_vectorizer

# TODO: Fix this class
class Word2Vec:
    def __init__(self, X, embedding_dim, save_model_dir):
        self.save_model_dir = save_model_dir
        try:
            self.word2vec = FastText.load(self.save_model_dir)
            print(f"Load model from {self.save_model_dir}")
        except:
            self.train_vocab(X, embedding_dim=embedding_dim)

    def __call__(self, *args, **kwargs):
        fasttext_model = FastText.load('./word2vec/fasttext_model.model')
        vocab_size = len(self.word_index) + 1
        output_dim = 32
        print(vocab_size)
        embedding_matrix = np.random.random((vocab_size, output_dim))
        for word, i in self.word_index.items():
            try:
                embedding_vector = fasttext_model.wv[word]
            except:
                print(word, 'not found')
            if embedding_vector is not None:
                embedding_matrix[i, :] = embedding_vector

        return embedding_matrix

    def train_vocab(self, X, embedding_dim):
        sentences = [sentence.split() for sentence in X]
        model = FastText(vector_size=embedding_dim, window=6, min_count=1, sentences=sentences, epochs=20)
        model.save(self.save_model_dir)
        
    def save_model(self, save_model_dir):
        try:
            with open(save_model_dir, 'wb') as file:
                pickle.dump(self, file)
            print(f'save successfully model with {save_model_dir}')

        except:
            print(f'can\'t save model with {save_model_dir}')

    @staticmethod
    def load_model(save_model_dir):
        with open(save_model_dir, 'rb') as file:
            model = pickle.load(file)

        return model


class BagOfWord:
    def __init__(self, X_train, X_test):
        self.ngram_range = (1, 1)
        self.X_train, self.X_test = X_train, X_test

        self.vectorizer = CountVectorizer(analyzer='word', input='content', ngram_range=self.ngram_range,
                                          max_features=None)

    def __call__(self, *args, **kwargs):
        X_train_bow = self.vectorizer.fit_transform(self.X_train).toarray()
        X_test_bow = self.vectorizer.transform(self.X_test).toarray()

        return X_train_bow, X_test_bow