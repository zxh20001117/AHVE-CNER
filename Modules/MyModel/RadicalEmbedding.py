import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP import logger
from fastNLP.embeddings.utils import get_embeddings
from transformers import AutoModel

from Utils.utils import RadicalVocab


class RadicalEmbedding(nn.Module):
    def __init__(self, tokenizer: AutoModel, embed_size: int = 64, char_emb_size: int = 64, char_dropout: float = 0,
                 dropout: float = 0,  activation='relu', requires_grad: bool = True,
                 include_word_start_end: bool = False):
        super().__init__()

        # activation function
        if isinstance(activation, str):
            if activation.lower() == 'relu':
                self.activation = F.relu
            elif activation.lower() == 'sigmoid':
                self.activation = F.sigmoid
            elif activation.lower() == 'tanh':
                self.activation = F.tanh
        elif activation is None:
            self.activation = lambda x: x
        elif callable(activation):
            self.activation = activation
        else:
            raise Exception(
                "Undefined activation function: choose from: [relu, tanh, sigmoid, or a callable function]")

        logger.info("Start constructing character vocabulary.")
        # 建立radical的词表
        self.radical_vocab = RadicalVocab(tokenizer)
        # exit() # 查看radical表
        self.char_pad_index = self.radical_vocab.radical_vocab.padding_idx
        logger.info(f"In total, there are {len(self.radical_vocab.radical_vocab)} distinct radical characters.")
        # 对vocab进行index
        max_radical_nums = self.radical_vocab.max_radical_nums

        if include_word_start_end:max_radical_nums += 2

        self.chars_to_radicals_embedding = torch.full((len(self.radical_vocab.char_tokenizer.vocab.items()),
                                                       max_radical_nums), fill_value=self.char_pad_index,
                                                      dtype=torch.long)

        self.word_lengths = torch.zeros(len(self.radical_vocab.char_tokenizer.vocab.items())).long()
        for word, index in self.radical_vocab.char_tokenizer.vocab.items():
            # if index!=vocab.padding_idx:  # 如果是pad的话，直接就为pad_value了。修改为不区分pad, 这样所有的<pad>也是同一个embed
            word = self.radical_vocab.char2radical(word)
            if include_word_start_end:
                word = ['<bow>'] + word + ['<eow>']
            self.chars_to_radicals_embedding[index, :len(word)] = \
                torch.LongTensor([self.radical_vocab.radical_vocab.to_index(radical) for radical in word])
            self.word_lengths[index] = len(word)
        # self.char_embedding = nn.Embedding(len(self.char_vocab), char_emb_size)
        self.char_embedding = get_embeddings((len(self.radical_vocab.radical_vocab), char_emb_size))
        self.char_embedding.weight.requires_grad = True

        self.embed_size = embed_size

        self.drop_word = nn.Dropout(char_dropout)
        self.dropout = nn.Dropout(dropout)
        self.requires_grad = requires_grad
        self.fc1 = nn.Linear(embed_size, 2*embed_size)
        self.fc2 = nn.Linear(2*embed_size, embed_size)

    def forward(self, words):
        r"""
        输入words的index后，生成对应的words的笔画二维张量表示。
        :param words: [batch_size, max_len]
        :return: [batch_size, max_len, radical_num, embed_size]
        """
        words = self.drop_word(words)
        chars = self.chars_to_radicals_embedding[words]  # batch_size x max_len x max_word_len
        masks = self.word_lengths[words]  # batch_size x max_len

        word_lengths = self.word_lengths[words]  # batch_size x max_len
        max_word_len = word_lengths.max()
        chars = chars[:, :, :max_word_len]
        chars = self.char_embedding(chars)  # batch_size x max_len x max_word_len x embed_size
        chars = self.fc1(chars)
        chars = self.dropout(chars)
        chars = self.activation(chars)
        chars = self.fc2(chars)

        return self.dropout(chars), masks
