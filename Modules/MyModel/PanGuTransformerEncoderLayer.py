import torch
import torch.nn as nn

from Modules.MyModel.AugMSA import AugMSA
from Modules.MyModel.SIFA_MLP import SIFA_MLP


class PanGuTransformerEncoderLayer(nn.Module):
    def __init__(self, emb_size, AugMSA_nums=8, dropout=0.15):
        super().__init__()
        self.layer_norm = nn.LayerNorm(emb_size)
        self.AugMSA = AugMSA(emb_size, AugMSA_nums)
        self.SIFA_MLP = SIFA_MLP(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # query = self.dropout(self.AugMSA(query, key, value)) + query
        query = self.layer_norm(query)
        query = self.dropout(self.AugMSA(query, key, value)) + value
        # query = self.dropout(self.SIFA_MLP(query)) + query
        query = self.dropout(self.SIFA_MLP(query)) + value
        query = self.layer_norm(query)
        return query


if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    emb_size = 768
    AugMSA_nums = 8
    query = torch.randn(batch_size, seq_len, emb_size).cuda()
    key = torch.randn(batch_size, seq_len, emb_size).cuda()
    encoder_layer = PanGuTransformerEncoderLayer(emb_size, AugMSA_nums).cuda()
    output = encoder_layer(query, key, key)
    print(output.shape)
