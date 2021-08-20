from torch.utils.data import Dataset
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CovidTweetsDataSet(Dataset):
  def __init__(self,train_data):
    self.train_data = train_data
  def __len__(self):
    return len(self.train_data)
  def __getitem__(self,idx):
    tokens = tokenizer.encode_plus(self.train_data.OriginalTweet[idx],padding='max_length',max_length=128,truncation=True,return_tensors='pt')
    label = self.train_data.Sentiment[idx]
    return tokens['input_ids'][0],tokens['token_type_ids'][0],tokens['attention_mask'][0],label