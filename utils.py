from collections import namedtuple

from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F


def slice_target(ids, len_, padding_idx=3):
    
    # ids: sos + sequence + eos
    len_ = len_-1
    idx = torch.arange(len(len_), dtype=torch.long)
    
    # input_ids: sos + sequence
    input_ids = ids.clone()
    input_ids[idx, len_] = padding_idx
    input_ids = input_ids[:,:-1]
    input_ = input_ids
    
    # output_ids: sequence + eos
    output_ids = ids.clone()
    output_ids = output_ids[:,1:]
    output = output_ids
    
    return input_, output


def train_batches(model, dataloader, criterion, optimizer, device):
    
    model.to(device)
    model.train()
        
    epoch_loss = 0
    for i, batch in enumerate(dataloader):
        
        id_, src_ids, src_len, tar_ids, tar_len  = batch
        # src_ids shape: (batch_size, fixed_seq_len)
        # src_len shape: (batch_size)
        # tar_ids shape: (batch_size, fixed_seq_len)
        # tar_len shape: (batch_size)
        
        src_input = (src_ids, src_len)
        tar_input, tar_output = slice_target(tar_ids, tar_len)
        
        optimizer.zero_grad()
                
        logits = model(src_input, tar_input)
        
        loss = criterion(logits, tar_output)
        loss.backward()
        
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)


def evaluate_batches(model, dataloader, criterion, device):
    
    model.to(device)
    model.eval()
    
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            
            id_, src_ids, src_len, tar_ids, tar_len  = batch
            # src_ids shape: (batch_size, fixed_seq_len)
            # src_len shape: (batch_size)
            # tar_ids shape: (batch_size, fixed_seq_len)
            # tar_len shape: (batch_size)

            src_input = (src_ids, src_len)
            tar_input, tar_output = slice_target(tar_ids, tar_len)
            
            logits = model(src_input, tar_input)
            # logits shape: (batch_size, tar_vocab_size, var_seq_len)
            
            batch_size, tar_vocab_size, pred_seq_len = logits.shape
            true_seq_len = tar_output.shape[1]
            
            if pred_seq_len > true_seq_len:
                pad = torch.ones((batch_size, pred_seq_len-true_seq_len), 
                                 dtype=torch.long, device=logits.device) * 3 # pad_id: 3
                tar_output = torch.cat([tar_output, pad],1)
                
            elif pred_seq_len < true_seq_len:
                tar_output = tar_output[:,:pred_seq_len]
            
            loss = criterion(logits, tar_output)
            epoch_loss += loss.item()
            
    return epoch_loss / len(dataloader)


def generate_sentence(src_sent, model, src_tokenizer, tar_tokenizer, device):
    
    model.to(device)
    
    model.eval()
    with torch.no_grad():
        src_ids = src_tokenizer.encode(src_sent)
        src_len = len(src_ids)
        
        src_ids = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
        src_len = torch.tensor([src_len], dtype=torch.long, device=device)
        src_input = (src_ids, src_len)
        
        logits = model(src_input, None)
        output = logits.argmax(1).squeeze().tolist()
        
        tar_sent = tar_tokenizer.decode(output)
        
    return tar_sent


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=5, verbose=False):
        """
        Args:
            patience (int): How long to wait agter last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improved.
                            Default: False
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        
        score = val_loss
        
        if self.best_score is None:
            self.best_score = score
            
        elif score > self.best_score:
            self.counter += 1 
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            torch.save(model.state_dict(), self.save_path)
            print("Saving the model to", self.save_path)
            self.best_score = score
            self.counter = 0
            
        return self.early_stop


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    print('Saving the model to', path)    
    torch.save(model.state_dict(), path)
    
    
def load_model(model, path):
    print('Loading the model from', path)
    model.load_state_dict(torch.load(path))
    return model