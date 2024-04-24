import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP import DataSet, logger

from fastNLP.embeddings.utils import get_embeddings

from typing import List

from transformers import AutoModel, AutoTokenizer

from Utils.paths import bert_path, root_path
from Utils.utils import PinyinVocab, pinyin_split


class CNNPinyinLevelEmbedding(nn.Module):
    def __init__(self, pinyin_vocab: PinyinVocab, feature_dim, char_emb_size: int = 128, pinyin_dropout=0.2,
                 dropout: float = 0.2, filter_nums: List[int] = (30, 30, 30), kernel_sizes: List[int] = (3, 2, 1),
                 pool_method: str = 'max', activation='relu', pre_train_char_embed: str = None,
                 requires_grad: bool = True):

        super().__init__()

        # for kernel in kernel_sizes: assert kernel % 2 == 1, "Only odd kernel is allowed."
        self.feature_dim = feature_dim
        assert pool_method in ('max', 'avg')
        self.pool_method = pool_method
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
        # 建立Pinyin的词表
        self.pinyin_vocab = pinyin_vocab
        # exit() # 查看Pinyin表
        self.pinyin_pad_index = self.pinyin_vocab.vocab.padding_idx
        self.pinyin_unk_index = self.pinyin_vocab.vocab.unknown_idx
        logger.info(f"In total, there are {len(self.pinyin_vocab.vocab)} distinct pinyin characters.")
        # 对vocab进行index
        # max_pinyin_nums = 3
        #
        # self.register_buffer('word_to_pinyin_embedding', torch.full((len(tokenizer.vocab.items()), max_pinyin_nums),
        #                                                             fill_value=self.pinyin_pad_index, dtype=torch.long))
        # self.word_to_pinyin_embedding = torch.full((len(tokenizer.vocab.items()), max_pinyin_nums),
        #                                            fill_value=self.pinyin_pad_index,
        #                                            dtype=torch.long)
        # self.word_lengths = torch.zeros(len(tokenizer.vocab.items())).long()
        #
        # for word, index in self.pinyin_vocab.char_tokenizer.vocab.items():
        #     # if index!=vocab.padding_idx:  # 如果是pad的话，直接就为pad_value了。修改为不区分pad, 这样所有的<pad>也是同一个embed
        #     pingyin = pinyin_split(word)  # 一个词的拼音
        #     if word in [tokenizer.unk_token, tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
        #         pingyin[0] = word
        #     self.word_to_pinyin_embedding[index, :3] = \
        #         torch.LongTensor([self.pinyin_vocab.vocab.to_index(p) for p in pingyin])
        #     self.word_lengths[index] = len(word)
        # self.char_embedding = nn.Embedding(len(self.char_vocab), char_emb_size)
        self.pinyin_embedding = get_embeddings((len(self.pinyin_vocab.vocab), char_emb_size))
        self.pinyin_embedding.weight.requires_grad = True

        self.filter_nums = filter_nums
        self.convs = nn.ModuleList([nn.Conv1d(
            self.pinyin_embedding.embedding_dim, filter_nums[i], kernel_size=kernel_sizes[i], bias=True,
            padding=kernel_sizes[i] // 2)
            for i in range(len(kernel_sizes))])
        self.fc = nn.Linear(sum(filter_nums), self.feature_dim)
        self.relu = nn.ReLU()
        self.drop_word = nn.Dropout(p=pinyin_dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.requires_grad = requires_grad

    def forward(self, pinyin_inputs):
        r"""
               输入words的index后，生成对应的words的拼音表示。

               :param words: [batch_size, max_len]
               :return: [batch_size, max_len, feature_dim]
        """
        # words = self.drop_word(words)
        # [batch_size, max_len, pinyin_nums = 3, char_emb_size]
        batch_size, max_len, _ = pinyin_inputs.size()
        # pinyins = self.word_to_pinyin_embedding[pinyin_inputs]  # batch_size x max_len x 3
        pinyin_nums = 3
        pinyins = self.pinyin_embedding(pinyin_inputs)  # batch_size x max_len x 3 x char_emb_size
        pinyins = self.drop_word(pinyins)

        reshaped_pinyins = pinyins.view(batch_size * max_len, pinyin_nums, -1).transpose(1, 2)
        # batch_size*max_len x char_emb_size x 3

        # conv_pinyins = [conv(reshaped_pinyins).transpose(1, 2).reshape(batch_size, max_len, pinyin_nums, -1)
        #                 for conv in self.convs]
        conv_pinyins = [conv(reshaped_pinyins).transpose(1, 2)
                        for conv in self.convs]
        for i in range(len(self.filter_nums)):
            conv_pinyins[i] = conv_pinyins[i].reshape(batch_size, max_len, -1, self.filter_nums[i])

        for i in range(len(conv_pinyins)):
            conv_pinyins[i], _ = torch.max(conv_pinyins[i], dim=-2)

        pinyins = torch.cat(conv_pinyins, dim=-1).contiguous() # B x max_len x sum(filters)

        # conv_pinyins = torch.cat(conv_pinyins, dim=-1).contiguous()  # B x max_len x max_word_len x sum(filters)
        # conv_pinyins = self.activation(conv_pinyins)
        # if self.pool_method == 'max':
        #     pinyins, _ = torch.max(conv_pinyins, dim=-2)  # batch_size x max_len x sum(filters)
        # else:
        #     pinyins = torch.sum(conv_pinyins, dim=-2) / conv_pinyins.size()[-1].float()
        pinyins = self.fc(pinyins)
        pinyins = self.relu(pinyins)
        self.dropout(pinyins)
        return pinyins


if __name__ == '__main__':
    data = DataSet({'sentences': [list('我是中国人不是美国人'), list('你是美国人')]})
    tokenizer = AutoTokenizer.from_pretrained(root_path + bert_path)
    pinyin_vocab = PinyinVocab(tokenizer)
    pinyinLevelEmbdder = CNNPinyinLevelEmbedding(pinyin_vocab, 1024, pinyin_dropout=0.2)

    test_sentences = [
        '我爱北京天安门',
        '我的家乡在美丽的东北松花江畔',
        '我'
    ]
    test_tokens = tokenizer.batch_encode_plus(test_sentences, add_special_tokens=True,
                                              padding='longest', return_tensors="pt")['input_ids']
    test_tokens = torch.LongTensor(test_tokens)

    pinyin_tokens = [
        [[10, 10, 10], [56, 24, 36], [3, 3, 3]],
        [[10, 10, 10], [56, 24, 36], [52, 28, 17]]
    ]
    import numpy as np
    pinyin_tokens = torch.tensor(np.array(pinyin_tokens))
    pinyin_embedding = pinyinLevelEmbdder(pinyin_tokens)
    print(pinyin_embedding.size())
