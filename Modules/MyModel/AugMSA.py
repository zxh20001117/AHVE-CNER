import torch
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AugMSA(nn.Module):
    def __init__(self, emb_size, AugMSA_nums=8):
        super().__init__()
        self.emb_size = emb_size
        self.AugMSA_nums = AugMSA_nums
        self.AugMSA_mat = nn.Parameter(torch.randn(AugMSA_nums, emb_size, emb_size), requires_grad=True).to(DEVICE)
        self.multi_head_attention = nn.MultiheadAttention(emb_size, 8)
        self.activation = nn.GELU()

    def forward(self, query, key, value):
        query = self.multi_head_attention(query, key, value, need_weights=False)[0]
        AugMSA = torch.matmul(torch.unsqueeze(value, 1), self.AugMSA_mat)
        AugMSA = self.activation(AugMSA)
        AugMSA = torch.sum(torch.permute(AugMSA, (0, 2, 3, 1)), dim=-1)
        query = query + AugMSA + value
        return query


if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    emb_size = 768
    AugMSA_nums = 8
    query = torch.randn(batch_size, seq_len, emb_size).to(DEVICE)
    key = torch.randn(batch_size, seq_len, emb_size).to(DEVICE)
    AugMSA = AugMSA(emb_size, AugMSA_nums).to(DEVICE)
    output = AugMSA(query, key, key)
    print(output.shape)
