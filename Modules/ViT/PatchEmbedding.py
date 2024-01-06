import torch.nn as nn
import torch
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, patch_dim, max_num_token=256):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = patch_dim
        self.max_num_token = max_num_token
        self.weight = nn.Parameter(torch.randn(self.patch_size * self.patch_size * 3, self.patch_dim),
                                   requires_grad=True)

    def forward(self, image):
        """
        :param image: [batch*seq_len, channel, height, width]
        :return patch_emb: [batch*seq_len, patch_num, patch_dim]
        """
        patch = F.unfold(image, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        patch_emb = torch.matmul(patch, self.weight)
        return patch_emb


if __name__ == "__main__":
    patchembedding = PatchEmbedding(2, 64)
    image = torch.randn(56, 3, 32, 32)
    patch_emb = patchembedding(image)
    print(patch_emb.shape)
