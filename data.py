from collections import namedtuple

import sentencepiece as spm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader



class Tokenizer(object):
    def __init__(self, model_path):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(model_path)
        
    def encode(self, sentence):
        """
        encode: a sentence to list of ids
        """
        return self.tokenizer.encode_as_ids(sentence)
    
    def decode(self, ids):
        """
        decode: list of ids to a sentence
        """
        return self.tokenizer.DecodeIds(ids)
    
    
class NMTDataset(Dataset):
    
    def __init__(self, sent_pairs, src_tokenizer, tar_tokenizer, device):
        super().__init__()
        
        self.src_tokenizer = src_tokenizer
        self.tar_tokenizer = tar_tokenizer
        self.device = device
        self.sent_pairs = sent_pairs # list of tuples
        
    @classmethod    
    def from_txt(cls, src_path, tar_path, src_tokenizer, tar_tokenizer, device):
        
        SentPair = namedtuple('SentPair', ['id','src_sent', 'tar_sent'])
        sent_pairs = list()
        with open(src_path, 'r') as src_file, open(tar_path, 'r') as tar_file:
            for id_, (src_sent, tar_sent) in enumerate(zip(src_file.readlines(), tar_file.readlines())):
                sent_pair = SentPair(id_, src_sent, tar_sent)
                sent_pairs.append(sent_pair)
                
        return cls(sent_pairs, src_tokenizer, tar_tokenizer, device)
        
    def __len__(self):
        
        return len(self.sent_pairs)
    
    def __getitem__(self, idx):
        
        return self.sent_pairs[idx]
    
    def _preprocess(self, sent_pair):
        
        id_, src_sent, tar_sent = sent_pair
        src_ids = self.src_tokenizer.encode(src_sent)
        tar_ids = [1]+self.tar_tokenizer.encode(tar_sent)+[2]
        src_len = len(src_ids)
        tar_len = len(tar_ids)
        
        return id_, src_ids, src_len, tar_ids, tar_len
    
    def _collate(self, batch):
        
        id_list = list()
        src_ids_list = list()
        src_len_list = list()
        tar_ids_list = list()
        tar_len_list = list()
        
        for sent_pair in batch:
            id_, src_ids, src_len, tar_ids, tar_len = self._preprocess(sent_pair)
            id_list.append(id_)
            src_ids_list.append(torch.tensor(src_ids, dtype=torch.long, device=self.device)) 
            tar_ids_list.append(torch.tensor(tar_ids, dtype=torch.long, device=self.device)) 
            src_len_list.append(src_len)
            tar_len_list.append(tar_len)
        
        id_ = id_list
        src_ids = nn.utils.rnn.pad_sequence(src_ids_list, batch_first=True, padding_value=3)
        tar_ids = nn.utils.rnn.pad_sequence(tar_ids_list, batch_first=True, padding_value=3)
        
        src_len = torch.tensor(src_len_list, dtype=torch.long, device=self.device)
        tar_len = torch.tensor(tar_len_list, dtype=torch.long, device=self.device)
        
        return id_, src_ids, src_len, tar_ids, tar_len
    
    def _split(self, dataset):
        
        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        return train_dataset, test_dataset
        
    def to_dataloader(self, batch_size=128, n_workers=6, split=True):
        res = None 
        if split:
            train_dataset, test_dataset = self._split(self)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=self._collate, 
                                          num_workers=n_workers)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=self._collate, 
                                         num_workers=n_workers)
            
            res = train_dataloader, test_dataloader
        else:
            dataloader = DataLoader(self, batch_size=batch_size, collate_fn=self._collate, num_workers=n_workers)
            
            res = dataloader
        
        return res