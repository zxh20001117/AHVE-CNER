from torch import nn
from Modules.LayerProcess import LayerProcess
from Modules.PositionWiseFeedForward import PositionWiseFeedForward
from Modules.MyModel.MultiHeadAttention import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads,
                 layer_preprocess_sequence, layer_postprocess_sequence,
                 dropout=None, scaled=True, ff_size=-1,
                 ff_activate='relu',
                 ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        self.scaled = scaled
        self.ff_activate = ff_activate
        self.dropout = dropout
        self.ff_size = ff_size

        self.layer_postprocess = LayerProcess(self.layer_postprocess_sequence, self.hidden_size, self.dropout['post'],
                                              )
        self.layer_postprocess1 = LayerProcess(self.layer_postprocess_sequence,self.hidden_size, self.dropout['post'],
                                               )

        self.attn = MultiHeadAttention(self.hidden_size, self.num_heads,
                                       scaled=self.scaled,
                                       attn_dropout=self.dropout['attn'],
                                       )

        self.ff = PositionWiseFeedForward([self.hidden_size, self.ff_size, hidden_size], self.dropout,
                                          ff_activate=self.ff_activate
                                          )

    def forward(self, query, key, value):

        output = self.attn(query, key, value)
        res = self.layer_postprocess(query, output)
        output = self.ff(res)
        output = self.layer_postprocess1(res, output)

        return output
