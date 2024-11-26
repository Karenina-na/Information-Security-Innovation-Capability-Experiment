import torch
import torch.nn as nn
import torch.optim as optim
from Models.Transformer.Coder.Modules import MultiHeadAttention, PoswiseFeedForwardNet, PositionalEncoding
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# Transformer Parameters
d_model = 128  # Embedding Size
d_ff = 1024  # FeedForward dimension
d_k = d_v = 32  # dimension of K(=Q), V
n_layers = 4  # number of Encoder of Decoder Layer
n_heads = 4  # number of heads in Multi-Head Attention
lr = 0.001  # learning rate
BatchSize = 128  # Batch size
Epoch = 100  # Epoch


# 一维数据分类模型
# position-wise feed forward net
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return self.layer_norm(output + residual)  # [batch_size, seq_len, d_model]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()

        self.d_k = d_k

    def forward(self, Q, K, V):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn = softmax(Q * K^T / sqrt(d_k))
        # context = softmax(attn * V)
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]

        attn = nn.Softmax(dim=-1)(scores)  # attention : [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(attn, V)  # context : [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

    def forward(self, input_Q, input_K, input_V):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k). \
            transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k). \
            transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v). \
            transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        context, attn = ScaledDotProductAttention(self.d_k).forward(Q, K, V)
        context = context.transpose(1, 2). \
            reshape(batch_size, -1, self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return self.layer_norm(output + residual), attn


# one decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        """
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = \
            self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn


class Decoder(nn.Module):
    def __init__(self, seq_size, n_layers, d_model, d_ff, d_k, d_v, n_heads, n_class):
        super(Decoder, self).__init__()
        self.embedding = nn.Sequential(
            # 卷积扩充维度 [batch, seq_len] -> [batch, seq_len, d_model]
            nn.Conv1d(seq_size, seq_size * d_model, 1)
        )
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, d_k, d_v, n_heads) for _ in range(n_layers)])
        self.classciation = nn.Sequential(
            nn.Linear(d_model, n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, dec_inputs):
        """
        dec_inputs: [batch_size, tgt_len]
        """
        dec_inputs = self.embedding(dec_inputs.unsqueeze(2)).reshape(dec_inputs.size(0), -1, d_model)
        # dec_inputs: [batch_size, tgt_len, d_model]
        dec_self_attns, dec_enc_attns = [], []
        dec_outputs = dec_inputs
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(dec_outputs)
            dec_self_attns.append(dec_self_attn)

        # dec_outputs: [batch_size, tgt_len, d_model] => [batch_size, d_model]
        dec_outputs = nn.AdaptiveAvgPool1d(1)(dec_outputs.permute(0, 2, 1)).squeeze(2)
        return self.classciation(dec_outputs)


if __name__ == '__main__':
    # 生成数据
    data = torch.tensor(np.random.randint(0, 100, (256, 34)), dtype=torch.float32).to('cuda:0')

    model = Decoder(34, n_layers, d_model, d_ff, d_k, d_v, n_heads, 4).to('cuda:0')
    result = model(data)
    print(result)
    pass
