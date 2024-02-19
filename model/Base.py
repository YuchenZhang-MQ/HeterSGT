import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

def sequence_mask(X, valid_len, value=0):
   
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e15)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class Config():
    def __init__(self):
        self.name = "model config"
    
    def print_config(self):
        for attr in self.attribute:
            print(attr)

class AdditiveAttention(nn.Module):
    
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
        
class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class MultiHeadAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs) -> None:
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
        values = self.transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.tensor(valid_lens)
            valid_lens = torch.repeat_interleave(valid_lens, repeats = self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
    
    def transpose_qkv(self, X, num_heads):
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        X = X.permute(0,2,1,3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X, num_heads):
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0,2,1,3)
        return X.reshape(X.shape[0], X.shape[1], -1)

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs) -> None:
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_output, **kwargs) -> None:
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.fcn1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.fcn2 = nn.Linear(ffn_num_hiddens, ffn_num_output)
    
    def forward(self, X):
        return self.fcn2(self.relu(self.fcn1(X)))

class EncoderBlock(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(query_size, key_size,
                                                value_size, num_hiddens,
                                                num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, walk_length, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        max_len = walk_length
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(
                10000,
                torch.arange(0, num_hiddens, 2, dtype=torch.float32) /
                num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class TransformerEncoder(nn.Module):
    def __init__(self, query_size, key_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, walk_length, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens

        self.pos_encoding = PositionalEncoding(num_hiddens, walk_length, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                EncoderBlock(query_size, key_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))
    def forward(self, X, valid_lens, *args):
        if args[0].case2 == "no":
    
            X = self.pos_encoding(X * math.sqrt(self.num_hiddens))

        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens=None)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

class FNclf(nn.Module):
    def __init__(self, input_dim, out_dim, dropout, **kwargs) -> None:
        super(FNclf, self).__init__(**kwargs)
        self.fcn1 = nn.Linear(input_dim, input_dim)
        self.fcn2 = nn.Linear(input_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h = self.fcn1(h).relu()
        h = self.dropout(h)
        h_emb = h
        return self.fcn2(h), h_emb

class FNclf1(nn.Module):
    def __init__(self, input_dim, out_dim, dropout, **kwargs) -> None:
        super(FNclf1, self).__init__(**kwargs)
        self.fcn1 = nn.Linear(input_dim, input_dim//2)
        self.fcn2 = nn.Linear(input_dim//2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h = self.fcn1(h).relu()
        h = self.dropout(h)
        return self.fcn2(h)
    
class Emb_Fuse_Layer(nn.Module):
    def __init__(self, input_dim, out_dim, dropout, **kwargs) -> None:
        super(Emb_Fuse_Layer, self).__init__(**kwargs)
        self.fcn1 = nn.Linear(input_dim, input_dim)
        self.fcn2 = nn.Linear(input_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Tanh()

    def forward(self, h):
        h = self.act(self.fcn1(h))
        h = self.dropout(h)
        return self.act(self.fcn2(h))
    
def genX(data, device):
    X = []
    walk_list, inner_list, type_list = data.walk_list.to(device), data.inner_list.to(device), data.type_list.to(device)
    for i, walk in enumerate(walk_list):
        x = []
        t_list = type_list[i]
        for j, id in enumerate(walk):
            
            if t_list[j] == 0:
                x.append(data.graph["news"].x[inner_list[i][j]]) 
            elif t_list[j] == 1:
                x.append(data.graph["entity"].x[inner_list[i][j]])
            elif t_list[j] == 2:
                x.append(data.graph["topic"].x[inner_list[i][j]])
              
        x = torch.stack(x).to(device)
        X.append(x)
    X = torch.stack(X)
    return X

