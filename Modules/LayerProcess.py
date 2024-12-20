import torch
from torch import nn


class LayerProcess(nn.Module):
    """
    Layer process module,decide whether to use (LayerNorm + Dropout + Residual) in Transformer
    """

    def __init__(self, process_sequence, hidden_size, dropout=0,
                 use_pytorch_dropout=True):
        super().__init__()
        self.use_pytorch_dropout = use_pytorch_dropout
        self.process_sequence = process_sequence.lower()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        if 'd' in self.process_sequence:
            self.dropout = nn.Dropout(dropout)
        if 'n' in self.process_sequence:
            self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, res, inp):
        output = inp
        # print(res.shape, inp.shape, self.hidden_size)
        for op in self.process_sequence:
            if op == 'a':
                output = res + inp
                assert not torch.isnan(output).any()
            if op == 'd':
                output = self.dropout(output)
            if op == 'n':
                output = self.layer_norm(output)
                assert not torch.isnan(output).any()

        return output
