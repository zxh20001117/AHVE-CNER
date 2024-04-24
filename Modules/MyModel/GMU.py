import torch
from torch import nn


class GatedMultimodalLayer(nn.Module):
    """
    ResGated Multimodal Layer based on 'Gated multimodal networks,
    Arevalo1 et al.' (https://arxiv.org/abs/1702.01992)
    """
    def __init__(self, emb_size):
        super(GatedMultimodalLayer, self).__init__()
        self.emb_size = emb_size
        self.hidden1 = nn.Linear(emb_size, emb_size, bias=False)
        self.hidden2 = nn.Linear(emb_size, emb_size, bias=False)
        self.hidden3 = nn.Linear(emb_size, emb_size, bias=False)
        self.hidden_sigmoid = nn.Linear(emb_size * 3, 3, bias=False)

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()
        self.softmax_f = nn.Softmax(dim=-1)

    def forward(self, x1, x2, x3):
        h1 = self.tanh_f(self.hidden1(x1)) + x1
        h2 = self.tanh_f(self.hidden1(x2)) + x2
        h3 = self.tanh_f(self.hidden1(x3)) + x3

        x = torch.cat((x1, x2, x3), dim=-1)
        z = self.sigmoid_f(self.hidden_sigmoid(x))
        z = self.softmax_f(z).unsqueeze(2)
        h = torch.stack((h1, h2, h3), dim=-1)
        output = torch.sum(h * z, dim=-1)
        return output


if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    emb_size = 768
    x1 = torch.randn(batch_size, seq_len, emb_size)
    x2 = torch.randn(batch_size, seq_len, emb_size)
    x3 = torch.randn(batch_size, seq_len, emb_size)
    GatedMultimodalLayer = GatedMultimodalLayer(emb_size)
    out = GatedMultimodalLayer(x1, x2, x3)
    print(out.shape)
