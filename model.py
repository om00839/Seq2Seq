import torch
from torch import nn
from torch.nn import functional as F


class EncLayer(nn.Module):
    """
    EncLayer (torch.nn.Module): Stacked Bidirectional LSTM Encoder Layer
    """
    def __init__(self, src_vocab_size, embedding_dim=256, 
                 hidden_dim=256, n_layers=2, bidirectional=True, 
                 padding_idx=3):
        super().__init__()
        
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        
        # Embedding
        self.lookup_table = nn.Embedding(src_vocab_size, embedding_dim, padding_idx)
        
        # Stacked Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, 
                            bidirectional=True, batch_first=True)
        
    def split_states(self, states):
        """
        Inputs
        ------
        states (tuple): (s, c)
            - s 
                shape: (n_layers*n_directions, batch_size, hidden_dim)
            - c
                shape: (n_layers*n_directions, batch_size, hidden_dim)
        
        Outputs
        -------
        states_splitted (list of tuples): [(s_0, c_0), ..., (s_L, c_L)] *L: n_layers
            - s_i
                shape: (batch_size, n_directions*hidden_dim)
            - c_i
                shape: (batch_size, n_directions*hidden_dim)
        """
        s, c = states
        s = s.transpose(0,1).reshape(-1, self.n_layers, self.n_directions*self.hidden_dim)
        c = c.transpose(0,1).reshape(-1, self.n_layers, self.n_directions*self.hidden_dim)
        
        states_splitted = [(s[:,n,:], c[:,n,:]) for n in range(self.n_layers)]
        
        return states_splitted
        
    def forward(self, src_input):
        """
        Inputs
        ------
        src_input (Tuple): (src_word indices, sequence lengths) 
            - word indices (LongTensor)
                shape: (batch_size, fixed_seq_len)
            - sequence lengths (LongTensor)
                shape: (batch_size)
            
        Outputs
        -------
        h (PackedSequence): source hidden states
            - data
                shape: (batch_size, fixed_seq_len, n_directions*hidden_dim)
            - lengths
                shape: (batch_size)
        
        final_states (list of tuples): [(s_n_0, c_n_0), ..., (s_n_L, c_n_L)] *L: n_layers
            - s_n_i
                shape: (batch_size, n_directions*hidden_dim)
            - c_n_i
                shape: (batch_size, n_directions*hidden_dim)
        """
        
        w, seq_len = src_input
        
        x = self.lookup_table(w) # word vectors
        x = nn.utils.rnn.pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)

        h, final_states = self.lstm(x) # encoder hidden states
        
        final_states = self.split_states(final_states)
        
        return h, final_states
    
    
class DecCell(nn.Module):
    """
    DecCell (torch.nn.Module): Stacked LSTM Decoder Cell
    """
    def __init__(self, attention, tar_vocab_size, 
                 embedding_dim=256, hidden_dim=256, n_layers=2, 
                 padding_idx=3, bidirectional=True):
        super().__init__()
        
        self.n_layers = n_layers
        n_directions = 2 if bidirectional else 1
        
        self.hidden_dim = hidden_dim
        self.enc_hidden_dim = n_directions * hidden_dim
        self.dec_hidden_dim = self.enc_hidden_dim
        
        
        # Embedding
        self.lookup_table = nn.Embedding(tar_vocab_size, embedding_dim, padding_idx)
        
        # Stacked LSTM
        self.lstm_cells = nn.ModuleList([nn.LSTMCell(embedding_dim, self.dec_hidden_dim)])
        for _ in range(n_layers-1):
            self.lstm_cells.append(nn.LSTMCell(self.dec_hidden_dim, self.dec_hidden_dim))
        
        self.attn = attention(self.hidden_dim, bidirectional)
        
        # Linear Classifier
        self.linear = nn.Linear(self.enc_hidden_dim + self.dec_hidden_dim, tar_vocab_size)
    
    def forward(self, w_t, h, states_t_1):
        """
        Inputs
        ------
        w_t (LongTensor): target word index at timestep t
            shape: (batch_size)
            
        h (PackedSequence): source hidden states
            - data
                shape: (batch_size, fixed_seq_len, n_directions*hidden_dim)
            - lengths
                shape: (batch_size)
             
        states_t_1 (list of tuples): [(s_t_1_0, c_t_1_0), ..., (s_t_1_L, c_t_1_L)] *L: n_layers
            - s_t_1_i
                shape: (batch_size, n_directions*hidden_dim)
            - c_t_1_i
                shape: (batch_size, n_directions*hidden_dim)
            
        Outputs
        -------
        logit_t (FloatTensor): target word logit at timestep t
            shape: (batch_size, tar_vocab_size)
             
        states_t (list of tuples): [(s_t_1_0, c_t_1_0), ..., (s_t_1_L, c_t_1_L)] *L: n_layers
            - s_t_i
                shape: (batch_size, n_directions*hidden_dim)
            - c_t_i
                shape: (batch_size, n_directions*hidden_dim)
        """
        
        y_t = self.lookup_table(w_t)
        # y_t shape: (batch_size, embedding_dim)
            
        states_t = list()
        for n in range(self.n_layers):
            lstm_cell_n = self.lstm_cells[n]
            states_t_1_n = states_t_1[n]
            input_n = y_t if n==0 else s_t
            
            s_t, c_t = lstm_cell_n(input_n, states_t_1_n)
            states_t.append((s_t, c_t))

        # s_t: decoder hidden state / c_t: decoder cell state
        # s_t shape: (batch_size, hidden_dim)
        # c_t shape: (batch_size, hidden_dim)

        z_t = self.attn(s_t, h) 
        # z_t: context vector
        # z_t shape: (batch_size, hidden_dim)

        sz_t = torch.cat([s_t,z_t],1)
        # sz_t: [s;z]
        # sz_t shape: (batch_size, hidden_dim + hidden_dim)

        logit_t = self.linear(sz_t)
        # logits_t shape: (batch_size, tar_vocab_size)
            
        return logit_t, states_t
    
    
class DecLayer(nn.Module): 
    """
    DecLayer (torch.nn.Module): Stacked LSTM Decoder Layer
    """
    def __init__(self, attention, tar_vocab_size, 
                 embedding_dim=256, hidden_dim=256, n_layers=2, 
                 bidirectional=True, padding_idx=3):
        super().__init__()
            
        # Stacked LSTM Cell
        self.dec_cell = DecCell(attention, tar_vocab_size, embedding_dim, 
                                hidden_dim, n_layers, padding_idx, 
                                bidirectional)
        
    def _train(self, tar_ids, h, init_states):
        """
        train: teacher forcing
        """
        
        w = tar_ids
        fixed_seq_len = w.shape[1]
        # w shape: (batch_size, fixed_seq_len)
        
        for t in range(fixed_seq_len):
                
            # w_t, states_t_1    
            w_t = w[:,t]
            states_t_1 = init_states if t == 0 else states_t
            # w_t shape: (batch_size, embedding_dim)
            
            # Decoder Cell
            logit_t, states_t = self.dec_cell(w_t, h, states_t_1)
            
            # logits
            logit_t = logit_t.unsqueeze(2)
            if t == 0:
                logits = logit_t
            else:
                logits = torch.cat([logits, logit_t], 2)
            # logit_t shape: (batch_size, tar_vocab_size, 1)
            # logits shape: (batch_size, tar_vocab_size, fixed_seq_len)
            
        return logits
        
    def _infer(self, h, init_states):
        """
        infer: greedy decoding
        """
        # initialize w_0 with sos_id(1)
        w_0 = torch.ones_like(h.unsorted_indices)
        # w_0 shape: (batch_size)
        
        for t in range(500):
            
            # w_t, states_t_1
            w_t = w_0 if t == 0 else output_t.squeeze(1)
            states_t_1 = init_states if t==0 else states_t
            
            # Decoder Cell
            logit_t, states_t = self.dec_cell(w_t, h, states_t_1)
            
            # logits
            logit_t = logit_t.unsqueeze(2)
            if t == 0:
                logits = logit_t
            else:
                logits = torch.cat([logits, logit_t], 2)
            # logit_t shape: (batch_size, 1, tar_vocab_size)
            # logits shape: (batch_size, fixed_seq_len, tar_vocab_size)
            
            # outputs
            output_t = logit_t.argmax(1)
            if t == 0:
                outputs = output_t
            else:
                outputs = torch.cat([outputs, output_t], 1)
            # output_t shape: (batch_size, 1)
            # outputs shape: (batch_size, t+1)
            
            # stop iteration if all batches have eod_id(2)
            if (outputs == 2).any(1).all():
                break
                
        return logits
        
    def forward(self, tar_ids, h, init_states):
        """
        Inputs
        ------
        tar_ids (tuple): word indices
            - word indices (LongTensor)
                shape: (batch_size, batch_max_seq_len)
                
        h (PackedSequence): source hidden states
            - data:
                shape: (batch_size, fixed_seq_len, n_directions*hidden_dim)
            - lengths:
                shape: (batch_size)
        
        init_states (tuple): (s_0, c_0)
            - s_0 
                shape: (batch_size, n_layers*n_directions, hidden_dim)
            - c_0
                shape: (batch_size, n_layers*n_directions, hidden_dim)
            
        Outputs
        -------
        logits (FloatTensor): 
            shape: (batch_size, tar_vocab_size, fixed_seq_len)
        
        """
        
        # init logits, outputs
        logits = None
        outputs = None
        if self.training:
            logits = self._train(tar_ids, h, init_states)
        else:
            logits = self._infer(h, init_states)
            
        return logits
    
    
class BaseAttention(nn.Module):
    
    def __init__(self, hidden_dim, bidirectional):
        super().__init__()
        
        n_directions = 2 if bidirectional == True else 1
        
        self.enc_hidden_dim = n_directions * hidden_dim
        self.dec_hidden_dim = self.enc_hidden_dim
        
    def get_mask(self, h, seq_len):
        
        batch_size, padded_seq_len, hidden_dim = h.shape
        mask = torch.zeros((batch_size, padded_seq_len), dtype=torch.bool)
        for i in range(batch_size):
            mask[i,seq_len[i]:] = 1
            
        return mask
    
    def compute_attn_scores(self, s_t, h):
        raise NotImplementedError
    
    def forward(self, s_t, h):
        """
        Inputs
        ------
        s_t (FloatTensor): a target hidden state at time step t
            shape: (batch_size, hidden_dim)
            
        h (PackedSequence): source hidden states
            - data:
                shape: (packed_seq_len, n_directions*hidden_dim)
            - lengths:
                shape: (batch_size)

        Outputs
        -------
        z (FloatTensor): context vector
            shape: (batch_size, hidden_dim) or (batch_size, n_directions*hidden_dim)
        """
        
        h, seq_len = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        # h shape: (batch_size, padded_seq_len, n_directions*hidden_dim)
        # seq_len: source sequence lengths
        
        e = self.compute_attn_scores(s_t, h) # attention scores
        mask = self.get_mask(h, seq_len) 
        e[mask] = -float('Inf') # for masked softmax
        # e shape: (batch_size, padded_seq_len)
        
        a = F.softmax(e, 1) # attention probabilities
        # a shape: (batch_size, padded_seq_len)
        
        z = (a.unsqueeze(2)*h).sum(1) # context vector
        # z shape: (batch_size, hidden_dim)
        
        return z
    
    
class DotProdAttention(BaseAttention):
    def __init__(self, hidden_dim=256, bidirectional=True):
        super(DotProdAttention, self).__init__(hidden_dim, bidirectional)
        
    def compute_attn_scores(self, s_t, h):
        # s_t shape: (batch_size, dec_hidden_dim)
        # h shape: (batch_size, padded_seq_len, enc_hidden_dim)
        
        e = (s_t.unsqueeze(1) @ h.transpose(1,2)).squeeze(2)
        # e shape: (batch_size, padded_seq_len)
        
        return e
    
    
class MulAttention(BaseAttention):
    def __init__(self, hidden_dim=256, bidirectional=True):
        super(MulAttention, self).__init__(hidden_dim, bidirectional)
        self.W = nn.Parameter(torch.randn((self.dec_hidden_dim, self.enc_hidden_dim)*0.01))
        
    def compute_attn_scores(self, s_t, h):
        # s_t shape: (batch_size, hidden_dim)
        # h shape: (batch_size, padded_seq_len, n_directions*hidden_dim)
        
        e = ((s_t.unsqueeze(1) @ self.W) @ h.transpose(1,2)).squeeze(2)
        # e shape: (batch_size, padded_seq_len)
        
        return e
    
    
class AddAttention(BaseAttention):
    def __init__(self, hidden_dim=256, bidirectional=True):
        super(AddAttention, self).__init__(hidden_dim, bidirectional)
        
        self.W_s = nn.Parameter(torch.randn((self.dec_hidden_dim, self.dec_hidden_dim), dtype=torch.float)*0.01)
        self.W_h = nn.Parameter(torch.randn((self.enc_hidden_dim, self.dec_hidden_dim), dtype=torch.float)*0.01)
        self.v = nn.Parameter(torch.randn((self.dec_hidden_dim,1), dtype=torch.float)*0.01)
        
    def compute_attn_scores(self, s_t, h):
        # s_t shape: (batch_size, dec_hidden_dim)
        # h shape: (batch_size, padded_seq_len, enc_hidden_dim)
        
        e = (torch.tanh((s_t.unsqueeze(1) @ self.W_s) + (h @ self.W_h)) @ self.v).squeeze(2)
        # e shape: (batch_size, padded_seq_len)
        
        return e
    
    
class Configuration:
    def __init__(self, src_vocab_size, tar_vocab_size, 
                 attention, embedding_dim=256, hidden_dim=256, 
                 n_layers=2, bidirectional=True):
        
        self.src_vocab_size = src_vocab_size
        self.tar_vocab_size = tar_vocab_size
        
        self.attention = attention
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        
class Seq2SeqWithAttn(nn.Module):
    
    def __init__(self, config):
        
        super().__init__()
        
        self.encoder = EncLayer(config.src_vocab_size, config.embedding_dim, config.hidden_dim, 
                                config.n_layers, config.bidirectional)
        
        self.decoder = DecLayer(config.attention, config.tar_vocab_size, config.embedding_dim, 
                                config.hidden_dim, config.n_layers, config.bidirectional)
        
    def forward(self, src_input, tar_ids):
        """
        Inputs
        ------
        src_input (Tuple): (src_word indices, sequence lengths) 
            - word indices (LongTensor)
                shape: (batch_size, fixed_seq_len)
            - sequence lengths (LongTensor)
                shape: (batch_size)
                
        tar_ids (tuple): word indices 
            - word indices (LongTensor)
                shape: (batch_size, batch_max_seq_len)

        Outputs
        -------
        logits (FloatTensor): 
            shape: (batch_size, tar_vocab_size, fixed_seq_len)
        """
        
        h, enc_final_states = self.encoder(src_input)
        
        dec_init_states = enc_final_states
        logits = self.decoder(tar_ids, h, dec_init_states)
        
        return logits