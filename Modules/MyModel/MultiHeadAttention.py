import math

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    """
    attention 计算公式：
        attention(Q, K, V) = softmax((Q1K^T) + Q2K'^T)/sqrt(d_k))V
        其中 Q1 Q2 为 query 加上 可学习的位置编码参数
        K为原始的key， K'为相对位置编码的key
    """
    def __init__(self, hidden_size, num_heads,
                 scaled=True,
                 attn_dropout=None,
                 use_pytorch_dropout=True):
        super().__init__()
        self.use_pytorch_dropout = use_pytorch_dropout
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled

        assert (self.per_head_size * self.num_heads == self.hidden_size)

        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_r = nn.Linear(self.hidden_size, self.hidden_size)
        self.u = nn.Parameter(torch.Tensor(self.num_heads, self.per_head_size))

        self.dropout = nn.Dropout(attn_dropout)

        # self.randomAttention = nn.Parameter(torch.empty(1, self.num_heads, self.max_seq_len, self.max_seq_len), requires_grad=True)
        #
        # nn.init.kaiming_normal_(self.randomAttention, a=math.sqrt(5))

    def forward(self, query, key, value):
        """
        :param query: 查询 [batch, seq_len, hidden_size]
        :param key: 查询键 [batch, seq_len, hidden_size]
        :param value: 值 [batch, seq_len, hidden_size]
        :return:
        """

        query = self.w_q(query)

        value = self.w_v(value)

        batch = key.size(0)

        # batch * seq_len * n_head * d_head
        key = torch.reshape(key, [batch, key.size(1), self.num_heads, self.per_head_size])
        query = torch.reshape(query, [batch, query.size(1), self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, value.size(1), self.num_heads, self.per_head_size])

        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        key = key.transpose(-1, -2)

        # 1 * n_head * 1 * d_head
        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)
        # batch * n_head * seq_len * d_head
        query_and_u_for_c = query + u_for_c
        # batch * n_head * seq_len * seq_len
        attn_score_raw = torch.matmul(query_and_u_for_c, key)

        if self.scaled:
            attn_score_raw_masked = attn_score_raw / math.sqrt(self.per_head_size)

        attn_score = F.softmax(attn_score_raw_masked, dim=-1)
        attn_score = self.dropout(attn_score)

        # batch * n_head * seq_len * d_head
        value_weighted_sum = torch.matmul(attn_score, value)

        # batch * n_head * seq_len * d_head -> batch * seq_len * n_head * d_head -> batch * seq_len * hidden_size
        result = value_weighted_sum.transpose(1, 2).contiguous(). \
            reshape(batch, query.size(2), self.hidden_size)

        return result


