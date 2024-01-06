import torch.nn as nn
import torch
import torch.nn.functional as F

from Modules.ViT.PatchEmbedding import PatchEmbedding


class VisionTransformer(nn.Module):
    def __init__(self, patch_size, patch_dim, max_num_token=256, num_heads=4, num_layers=6):
        super().__init__()
        self.patch_dim = patch_dim
        self.max_num_token = max_num_token
        self.patch_embedder = PatchEmbedding(patch_size, patch_dim, max_num_token)
        self.positional_embedding = torch.randn(self.max_num_token, self.patch_dim, requires_grad=True)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.patch_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, image):
        """
        :param image: [batch*seq_len, channel, height, width]
        :return encoder_output: [batch*seq_len, patch_num, patch_dim]
        """
        bs, ic, ih, iw = image.shape
        patch_emb = self.patch_embedder(image)
        # positional embedding
        patch_emb += torch.tile(self.positional_embedding, [bs, 1, 1])

        # pass to transformer encoder
        encoder_output = self.transformer_encoder(patch_emb)
        return encoder_output


if __name__ == "__main__":
    vit = VisionTransformer(2, 64)
    image = torch.randn(1, 3, 32, 32)
    patch_emb = vit(image)
    print(patch_emb.shape)
