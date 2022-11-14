# -*- coding: utf-8 -*-
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15
from layers.dynamic_rnn import DynamicLSTM


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn


class MultiHeadAttention(nn.Module):

    def __init__(self, head, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(2)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        # mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.head, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class MLGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(MLGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.pos_emb = nn.Embedding(opt.pos_size, opt.pos_dim, padding_idx=0) if opt.pos_dim > 0 else None        # POS emb
        self.text_lstm = DynamicLSTM(opt.embed_dim+opt.pos_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, opt.hidden_dim)
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)
        self.in_drop = nn.Dropout(0.1)
        self.rnn_drop = nn.Dropout(opt.rnn_drop)
        self.gcn_drop = nn.Dropout(opt.gcn_drop)
        self.attn = MultiHeadAttention(opt.attention_heads, 2*opt.hidden_dim)
        self.affine1 = nn.Parameter(torch.Tensor(opt.hidden_dim, opt.hidden_dim))
        self.affine2 = nn.Parameter(torch.Tensor(opt.hidden_dim, opt.hidden_dim))

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return weight*x

    def position_weight_new(self, x, height, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        aspect_length = aspect_len.cpu().numpy()
        for i in range(batch_size):
            alpha = []
            tree_deep = height[i][0]  # tree deep
            asp_post = aspect_double_idx[i][0]
            length = int(text_len[i])
            assert len(height[i][1:]) == int(text_len[i])
            for high in height[i][1:]:
                alpha.append(1 - abs(height[i][asp_post + 1] - high) / tree_deep)
            alpha[aspect_double_idx[i][0]-1] = -1e10
            alpha = (length*entmax15(torch.tensor(alpha), dim=-1)).to(self.opt.device)
            alpha[aspect_double_idx[i][0]-1] = 1
            alpha=alpha.unsqueeze(1)
            x[i][:length] = x[i][:length] * alpha
        return x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return mask*x

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, pos_indices, adj, asp_post, height = inputs
        pad_mask = (text_indices != 0).unsqueeze(-2)
        mask_ = (torch.zeros_like(text_indices) != text_indices).float().unsqueeze(-1)
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_real_len = torch.sum(aspect_indices != 0, dim=-1)
        aspect_len = torch.ones(self.opt.batch_size).to(self.opt.device)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), left_len.unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        # aspect_avg_emb = torch.zeros(self.opt.batch_size, 300)
        for i in range(text_indices.shape[0]):
            aspect = aspect_indices[i][0:aspect_real_len[i]]
            text[i][asp_post[i]] = torch.mean(self.embed(aspect), dim=0)

        embs = [text]
        if self.opt.pos_dim > 0:
            embs += [self.pos_emb(pos_indices)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(embs, text_len)
        text_out = self.position_weight_new(text_out, height, aspect_double_idx, text_len, aspect_len)
        text_out = self.rnn_drop(text_out)

        #******Self Attention********
        attn_tensor = self.attn(text_out, text_out, pad_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        # outputs_dep = None
        adj_sem = None

        # * Average Multi-head Attention matrixes
        for i in range(self.opt.attention_heads):
            if adj_sem is None:
                adj_sem = attn_adj_list[i]
            else:
                adj_sem += attn_adj_list[i]
        adj_sem = adj_sem / self.opt.attention_heads

        for j in range(adj_sem.size(0)):
            adj_sem[j] -= torch.diag(torch.diag(adj_sem[j]))
            adj_sem[j] += torch.eye(adj_sem[j].size(0)).to(self.opt.device)
        adj_sem = mask_ * adj_sem

        # ************SynGCN*************
        x = F.relu(self.gc1(self.position_weight_new(text_out, height, aspect_double_idx, text_len, aspect_len), adj))
        # x = F.relu(self.gc1(text_out, adj))
        x = F.relu(self.gc2(x, adj))
        # ************SemGCN*************
        x_sem = F.relu(self.gc1(text_out, adj_sem))
        x_sem = F.relu(self.gc2(x_sem, adj_sem))

        # * mutual Biaffine module
        A1 = F.softmax(torch.bmm(torch.matmul(x, self.affine1), torch.transpose(x_sem, 1, 2)), dim=-1)
        A2 = F.softmax(torch.bmm(torch.matmul(x_sem, self.affine2), torch.transpose(x, 1, 2)), dim=-1)
        x, x_sem = torch.bmm(A1, x_sem), torch.bmm(A2, x)
        x = self.gcn_drop(x)
        x_sem = self.gcn_drop(x_sem)
        x = self.mask(x, aspect_double_idx)
        x_sem = self.mask(x_sem, aspect_double_idx)

        output = x.sum(dim=1)
        output_sem = x_sem.sum(dim=1)
        final_output = torch.cat((output, output_sem), dim=-1)

        logits = self.fc(final_output)
        return logits
