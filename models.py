from transformers import BertModel
import torch.nn as nn

class BertBaseUnCased(nn.Module):
  def __init__(self):
    super(BertBaseUnCased,self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.drop = nn.Dropout(0.3)
    self.linear = nn.Linear(768,5)
  def forward(self,ids,type_ids,mask):
    x = self.bert(ids,attention_mask = mask,token_type_ids = type_ids)['pooler_output']
    x = self.drop(x)
    x = self.linear(x)
    return x

