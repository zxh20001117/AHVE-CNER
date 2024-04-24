import torch
import torch.nn as nn

from Modules.MyModel.PanGuTransformerEncoderLayer import PanGuTransformerEncoderLayer


class PanGuTransformerEncoder(nn.Module):
    def __init__(self, emb_size=768, AugMSA_nums=8, dropout=0.15):
        super().__init__()
        self.emb_size = emb_size
        self.AugMSA_nums = AugMSA_nums
        self.dropout = dropout
        self.encoder_layer = PanGuTransformerEncoderLayer(emb_size=self.emb_size, AugMSA_nums=self.AugMSA_nums,
                                                          dropout=self.dropout)

    def forward(self, query, key, value):
        return self.encoder_layer(query, key, value)
