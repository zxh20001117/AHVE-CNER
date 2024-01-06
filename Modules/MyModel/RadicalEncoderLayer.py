import json

import torch
import torch.nn as nn
from fastNLP import seq_len_to_mask

from Modules.MyModel.TransformerEncoderLayer import TransformerEncoderLayer


class RadicalEncoderLayer(nn.Module):
    def __init__(self, patch_size, patch_dim, layer_preprocess_sequence, layer_postprocess_sequence, dropout=None,
                 scale=True, nums_head=4, ff_size=-1, ff_activate='relu'):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = patch_dim
        self.scaled = scale
        self.ff_activate = ff_activate
        self.ff_size = ff_size
        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        self.dropout = dropout

        self.transformer_layer = TransformerEncoderLayer(self.patch_dim, nums_head,
                                                         self.layer_preprocess_sequence,
                                                         self.layer_postprocess_sequence,
                                                         self.dropout, self.scaled, self.ff_size,
                                                         ff_activate=self.ff_activate
                                                         )
        self.GRU = nn.GRU(self.patch_dim, self.patch_dim, batch_first=True)

    def forward(self, patch_emb, radical_emb, mask):
        """
        :param patch_emb: [batch*seq_len, patch_num, patch_dim]
        :param radical_emb: [batch*seq_len, radical_num, radical_dim]
        :return: [batch*seq_len, patch_dim]
        """
        # [batch*seq_len, radical_num, radical_dim]
        radical_emb = self.transformer_layer(radical_emb, patch_emb, patch_emb)
        mask = seq_len_to_mask(mask, max_len=radical_emb.shape[1])  # [batch*seq_len, radical_num]
        radical_emb = radical_emb.masked_fill(mask.unsqueeze(-1) == 0, 0)
        output, hidden = self.GRU(radical_emb)
        hidden = hidden.squeeze(0)

        return hidden


if __name__ == "__main__":
    config = json.load(
        open("../../config.json", 'r', encoding='utf-8')
    )
    lattice_transformer = RadicalEncoderLayer(patch_size=config['vit']['patch_size'],
                                              patch_dim=config['vit']['patch_dim'],
                                              dropout=config['dropout'],
                                              layer_preprocess_sequence=config['layer_preprocess_sequence'],
                                              layer_postprocess_sequence=config['layer_postprocess_sequence'],
                                              ff_size=config['ff_size']
                                              )
    patch_emb = torch.randn(3, 256, 64)
    radical_emb = torch.randn(3, 6, 64)
    mask = torch.LongTensor([4, 5, 6])
    output = lattice_transformer(patch_emb, radical_emb, mask)
    print(output.shape)
