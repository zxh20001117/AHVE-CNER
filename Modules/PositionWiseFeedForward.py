from torch import nn


class PositionWiseFeedForward(nn.Module):
    '''
        self-attention 层 之后的 逐位前馈网络，对每一个位置的向量使用同一个前馈神经网络进行同样的变换
    '''
    def __init__(self, sizes, dropout=None, ff_activate='relu',
                 use_pytorch_dropout=True):
        super().__init__()
        self.use_pytorch_dropout = use_pytorch_dropout
        self.num_layers = len(sizes) - 1
        for i in range(self.num_layers):
            setattr(self, 'w' + str(i), nn.Linear(sizes[i], sizes[i + 1]))

        self.dropout = nn.Dropout(dropout['ff'])
        self.dropout_2 = nn.Dropout(dropout['ff_2'])

        if ff_activate == 'relu':
            self.activate = nn.ReLU(inplace=True)
        elif ff_activate == 'leaky':
            self.activate = nn.LeakyReLU(inplace=True)

    def forward(self, inp):
        output = inp
        for i in range(self.num_layers):
            if i != 0:
                output = self.activate(output)
            w = getattr(self, 'w' + str(i))

            output = w(output)
            if i == 0:
                output = self.dropout(output)
            if i == 1:
                output = self.dropout_2(output)

        return output
