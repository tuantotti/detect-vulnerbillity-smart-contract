import torch.nn as nn
import torch

class Branch(nn.Module):
  def __init__(self, input_size, hidden_size, dropout, num_outputs):
    super(Branch, self).__init__()

    self.dense1 = nn.Linear(input_size, hidden_size)
    self.batchnorm1 = nn.BatchNorm1d(hidden_size)
    self.dropout = nn.Dropout(p=dropout)
    self.dense2 = nn.Linear(hidden_size, num_outputs)

  def forward(self, x):
    out_dense1 = self.dense1(x)
    out_batchnorm1 = self.batchnorm1(out_dense1)
    out_dropout = self.dropout(out_batchnorm1)
    out_dense2 = self.dense2(out_dropout)

    return out_dense2

class Escort(nn.Module):
  def __init__(self, vocab_size, embedd_size, rnn_hidden_size, n_layers, num_classes, method_type, bidirectional, is_multibranches, num_auxiliary=None, auxiliary_feature_length=None):
    super(Escort, self).__init__()
    
    self.word_embeddings = nn.Embedding(vocab_size, embedd_size, padding_idx=0)
    self.bidirectional = bidirectional
    self.is_multibranches = is_multibranches
    self.rnn_hidden_size = rnn_hidden_size
    if method_type=='GRU':
        self.rnn = nn.GRU(embedd_size, self.rnn_hidden_size, num_layers=n_layers, batch_first=True, bidirectional=self.bidirectional)
    else:
        self.rnn = nn.LSTM(embedd_size, self.rnn_hidden_size, num_layers=n_layers, batch_first=True, bidirectional=self.bidirectional)
    
    branch_feature_size = self.rnn_hidden_size
    if num_auxiliary is not None and auxiliary_feature_length is not None:
        if num_auxiliary > 2 or num_auxiliary < 0:
            raise ValueError('num_auxiliary must be in (0,1,2)')
        branch_feature_size = rnn_hidden_size + num_auxiliary*auxiliary_feature_length
    
    if self.is_multibranches:
        self.branches = nn.ModuleList([Branch(branch_feature_size, 128, 0.2, 1) for _ in range(num_classes)])
    else:
        self.branch = Branch(branch_feature_size, 128, 0.2, num_classes)
    
    
    self.sigmoid = nn.Sigmoid()

  def forward(self, sequence, tfidf_features=None,word2vec=None):
    # print("Input to Escort:", sequence.shape)
    embeds = self.word_embeddings(sequence)
    rnn_out, _ = self.rnn(embeds)
    
    if self.bidirectional:
        rnn_out = (rnn_out[:, :, :self.rnn_hidden_size] + rnn_out[:, :, self.rnn_hidden_size:])
    
    last_hidden_state = rnn_out[:, -1, :]
    if tfidf_features is not None:
        last_hidden_state = torch.cat([last_hidden_state, tfidf_features], dim=1)
    if word2vec is not None:
        last_hidden_state = torch.cat([last_hidden_state, word2vec], dim=1)
        
    if self.is_multibranches:
        output_branches = [branch(last_hidden_state) for branch in self.branches]
        out_branch = torch.cat(output_branches, dim=1)
    else:
        out_branch = self.branch(last_hidden_state)
    outputs = self.sigmoid(out_branch)
    return outputs
  
class BaseModel(nn.Module):
    def __init__(self, original_model, num_classes):
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
        self.original_model = original_model
        self.branches = nn.ModuleList([Branch(768, 128, 64, 0.1, 2) for _ in range(num_classes)])
        self.activation = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out_bert = self.original_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooler_out = out_bert.pooler_output
        output_branches = [branch(pooler_out) for branch in self.branches]
        outputs = [self.activation(out_branch) for out_branch in output_branches]

        # apply softmax function for each branch
        out_soft = [self.activation(out) for out in outputs]
        out_soft_max_indices = [torch.argmax(out, dim=1) for out in out_soft]
        out_soft_max_indices = torch.stack(out_soft_max_indices, dim=1)

        return out_soft, out_soft_max_indices
    
class WisdomNet(nn.Module):
  def __init__(self, trained_model):
    super(WisdomNet, self).__init__()
    self.numclasses = trained_model.num_classes
    self.bert = trained_model.original_model

    num_outputs = 3 # add a reject label --> num_outputs = old_outputs + 1
    self.new_branches = nn.ModuleList([Branch(768, 128, 64, 0.1, num_outputs) if i == 1 else Branch(768, 128, 64, 0.1, 2) for i in range(self.numclasses)])

    for i in range(self.numclasses):
      old_branch = trained_model.branches[i]
      new_branch = self.new_branches[i]
      self.copy_weight_branch(old_branch, new_branch)
      new_branch.dense3.weight.data[:2].copy_(old_branch.dense3.weight.data.clone())
      new_branch.dense3.bias.data[:2].copy_(old_branch.dense3.bias.data.clone())

    self.softmax = nn.Softmax(dim=1)

  def copy_weight_branch(self, old_branch, new_branch, name_except='dense3'):
    for name, param in old_branch.named_parameters():
      if name_except not in name:
        name_layer = name.split('.')[0]
        name_attr_layer = name.split('.')[1]
        new_model_param = getattr(new_branch, name_layer)
        if name_attr_layer == 'weight':
          new_model_param.weight.data.copy_(param.data.clone())
        if name_attr_layer == 'bias':
          new_model_param.bias.data.copy_(param.data.clone())

  def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
    out_bert = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    pooler_out = out_bert.pooler_output
    output_branches = [branch(pooler_out) for branch in self.new_branches]

    # apply softmax function for each branch
    out_soft = [self.softmax(out) for out in output_branches]
    out_soft_max_indices = [torch.argmax(out, dim=1) for out in out_soft]
    out_soft_max_indices = torch.stack(out_soft_max_indices, dim=1)

    return out_soft, out_soft_max_indices    