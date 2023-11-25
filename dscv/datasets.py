import torch
from dscv.utils.process_text import pad_sequences
from torch.utils.data import Dataset
from dataclasses import dataclass
from dscv.utils.feature_extraction_utils import Word2Vec
    
class AuxiliaryOpcodeData(Dataset):
    """
    Dataset for input
    tfidf_vectorizer!=None or word2vec!=None for auxiliary input settings
    """
    def __init__(self, X, y, tokenizer, max_len, tfidf_vectorizer=None, word2vec=None):
        self.tokenizer = tokenizer
        self.X = X
        self.targets = y
        self.max_len = max_len
        self.tfidf_vectorizer = tfidf_vectorizer
        self.word2vec = word2vec

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        text = str(self.X[index])

        tfidf = self.tfidf_vectorizer.transform([text])
        sequence = self.tokenizer.texts_to_sequences(texts=[text])
        ids = pad_sequences(sequence, maxlen=self.max_len)


        return {
            'ids': torch.tensor(ids.flatten(), dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float),
            'tfidf': torch.tensor(tfidf.flatten(), dtype=torch.float),
        }
        
class OpcodeData(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.X = X
        self.targets = y
        self.max_len = max_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        text = str(self.X[index])

        inputs = self.tokenizer(
            text,
            None,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'index': index,
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }