import json

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from Modules.MyModel.RadicalEmbedding import RadicalEmbedding
from Modules.MyModel.RadicalEncoderLayer import RadicalEncoderLayer
from Modules.PositionalEncoding import PositionalEncoding
from Modules.ViT.Char2Image import Char2Image
from Modules.ViT.VisionTransformer import VisionTransformer
from Utils.paths import root_path, bert_path, ttf_paths


class RadicalEncoder(nn.Module):
    def __init__(self, patch_size, patch_dim, radical_dim, feature_dim, dropout, layer_preprocess_sequence,
                 layer_postprocess_sequence, ff_size, vit_max_num_token, vit_num_heads, vit_num_layers,
                 radical_num_heads, radical_num_layers, tokenizer):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = patch_dim
        self.radical_dim = radical_dim
        self.feature_dim = feature_dim

        self.radical_encoder_layer = RadicalEncoderLayer(patch_size=patch_size,
                                                         patch_dim=patch_dim,
                                                         dropout=dropout,
                                                         layer_preprocess_sequence=layer_preprocess_sequence,
                                                         layer_postprocess_sequence=layer_postprocess_sequence,
                                                         ff_size=ff_size)
        self.vit = VisionTransformer(self.patch_size, self.patch_dim, max_num_token=vit_max_num_token,
                                     num_heads=vit_num_heads, num_layers=vit_num_layers)
        self.radical_embedder = RadicalEmbedding(tokenizer=tokenizer, embed_size=self.radical_dim)
        self.radical2feature = nn.Linear(self.radical_dim, self.feature_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=radical_num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=radical_num_layers)
        # self.radical_positional_embedding = nn.Parameter(torch.Tensor(self.radical_dim), requires_grad=True)
        self.positional_encoding = PositionalEncoding(self.feature_dim, 0)

    def forward(self, token, image):
        """
        :param token: [batch, seq_len]
        :param image: [batch, seq_len, channel, height, width]
        :return: radical_emb: [batch, seq_len, emb_dim]
        """
        batch, seq_len, channel, height, width = image.shape
        image = image.view(batch * seq_len, channel, height, width)
        patch_emb = self.vit(image)  # [batch*seq_len, patch_num, patch_dim]
        assert not torch.isnan(patch_emb).any()
        radical_emb, mask = self.radical_embedder(token)  # [batch, seq_len, radical_num, radical_dim], [batch, seq_len]
        assert not torch.isnan(radical_emb).any()
        batch, seq_len, radical_num, radical_dim = radical_emb.shape

        # [batch*seq_len, radical_num, radical_dim]
        radical_emb = radical_emb.view(batch * seq_len, radical_num, radical_dim)
        assert not torch.isnan(radical_emb).any()
        mask = mask.view(batch * seq_len)
        # [batch*seq_len, patch_dim]

        radical_emb = self.radical_encoder_layer(patch_emb, radical_emb, mask)
        assert not torch.isnan(radical_emb).any()
        radical_emb = radical_emb.view(batch, seq_len, self.radical_dim)
        radical_emb = self.radical2feature(radical_emb)
        radical_emb = self.positional_encoding(radical_emb)
        radical_emb = self.transformer_encoder(radical_emb)

        return radical_emb


if __name__ == "__main__":
    config = json.load(
        open("../../config.json", 'r', encoding='utf-8')
    )

    patch_size = config['vit']['patch_size']
    patch_dim = config['vit']['patch_dim']
    radical_dim = config['vit']['patch_dim']
    dropout = config['dropout']
    layer_preprocess_sequence = config['layer_preprocess_sequence']
    layer_postprocess_sequence = config['layer_postprocess_sequence']
    ff_size = config['ff_size']
    vit_max_num_token = config['vit']['max_num_token']
    vit_num_heads = config['vit']['num_heads']
    vit_num_layers = config['vit']['num_layers']
    radical_num_heads = config['radical']['num_heads']
    radical_num_layers = config['radical']['num_layers']
    font_size = config['vit']['font_size']

    test_sentences = [
        '我爱北京天安门',
        '我的家乡在美丽的东北松花江畔',
        '我',
        '我爱',
    ]
    tokenizer = AutoTokenizer.from_pretrained(root_path + bert_path)
    test_tokens = tokenizer.batch_encode_plus(test_sentences, add_special_tokens=False,
                                              padding='longest', return_tensors="pt")['input_ids']
    max_word_len = max(len(test_sentence) for test_sentence in test_sentences)
    batch_size = len(test_sentences)
    char2img = Char2Image([root_path + ttf_path for ttf_path in ttf_paths], font_size)
    test_images = [torch.Tensor(char2img.get_images(test_sentence)) for test_sentence in test_sentences]


    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data


    img_mat = torch.zeros((batch_size, max_word_len, 3, font_size, font_size), dtype=torch.float32)
    img_inputs = fill(test_images, img_mat)

    radical_encoder = RadicalEncoder(patch_size=patch_size, patch_dim=patch_dim, radical_dim=radical_dim,
                                     feature_dim=1024,
                                     dropout=dropout, layer_preprocess_sequence=layer_preprocess_sequence,
                                     layer_postprocess_sequence=layer_postprocess_sequence, ff_size=ff_size,
                                     vit_max_num_token=vit_max_num_token, vit_num_heads=vit_num_heads,
                                     vit_num_layers=vit_num_layers, radical_num_heads=radical_num_heads,
                                     radical_num_layers=radical_num_layers, tokenizer=tokenizer)
    print(img_inputs.shape)
    radical_emb = radical_encoder(test_tokens, img_inputs)
    print(radical_emb.shape)
