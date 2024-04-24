import json

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import prettytable as pt

from Modules.ViT.Char2Image import Char2Image
from Utils import utils
from Utils.paths import root_path, ttf_paths, bert_path
from Utils.utils import LabelVocab, PinyinVocab

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, img_inputs, pinyin_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text):
        self.bert_inputs = bert_inputs
        self.img_inputs = img_inputs
        self.pinyin_inputs = pinyin_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.entity_text = entity_text

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
            torch.LongTensor(self.img_inputs[item]), \
            torch.LongTensor(self.pinyin_inputs[item]), \
            torch.LongTensor(self.grid_labels[item]), \
            torch.LongTensor(self.grid_mask2d[item]), \
            torch.LongTensor(self.pieces2word[item]), \
            torch.LongTensor(self.dist_inputs[item]), \
            self.sent_length[item], \
            self.entity_text[item]

    def __len__(self):
        return len(self.bert_inputs)


def process_bert(data, tokenizer, label_vocab, char2img, pinyin_vocab):
    bert_inputs = []
    img_inputs = []
    pinyin_inputs = []
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    sent_length = []

    for index, instance in enumerate(data):
        if len(instance['sentence']) == 0:
            continue

        # tokens = [tokenizer.tokenize(word) for word in instance['sentence']]
        # pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(instance['sentence'])
        # _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])
        _bert_inputs = np.array(_bert_inputs)
        _img_inputs = char2img.get_images(instance['sentence'])
        _pinyin_input = pinyin_vocab.sentence2pinyin(instance['sentence'])
        length = len(instance['sentence'])
        _grid_labels = np.zeros((length, length), dtype=np.int)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
        _dist_inputs = np.zeros((length, length), dtype=np.int)
        _grid_mask2d = np.ones((length, length), dtype=np.bool)

        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(instance['sentence']):
                if len(pieces) == 0:
                    continue
                pieces = [start, start + len(pieces)]
                _pieces2word[i, pieces[0]:pieces[-1]] = 1
                start += pieces[-1] - pieces[0]

        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k

        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19

        for entity in instance["ner"]:
            index = entity["index"]
            for i in range(len(index)):
                if i + 1 >= len(index):
                    break
                _grid_labels[index[i], index[i + 1]] = 1
            _grid_labels[index[-1], index[0]] = label_vocab.label_to_id(entity["type"])

        _entity_text = set([utils.convert_index_to_text(e["index"], label_vocab.label_to_id(e["type"]))
                            for e in instance["ner"]])

        sent_length.append(length)
        bert_inputs.append(_bert_inputs)
        img_inputs.append(_img_inputs)
        pinyin_inputs.append(_pinyin_input)
        grid_labels.append(_grid_labels)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)
        entity_text.append(_entity_text)

    return bert_inputs, img_inputs, pinyin_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


def collate_fn(data):
    bert_inputs, img_inputs, pinyin_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text = map(list, zip(*data))

    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    img_mat = torch.zeros((batch_size, max_tok, 3, 32, 32), dtype=torch.float32)
    img_inputs = fill(img_inputs, img_mat)
    pinyin_mat = torch.zeros((batch_size, max_tok, 3), dtype=torch.float32)
    pinyin_inputs = fill(pinyin_inputs, pinyin_mat)
    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)
    labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return bert_inputs, img_inputs, pinyin_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


def load_data_bert(config):
    with open(f"./data/{config['dataset']}/train.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(f"./data/{config['dataset']}/test.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    with open(f"./data/{config['dataset']}/dev.json", 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    font_size = config['vit']['font_size']
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    char2img = Char2Image([root_path + ttf_path for ttf_path in ttf_paths], font_size)
    pinyin_vocab = PinyinVocab(tokenizer)

    vocab = LabelVocab()
    train_ent_num = vocab.fill_vocab(train_data)
    test_ent_num = vocab.fill_vocab(test_data)
    dev_ent_num = vocab.fill_vocab(dev_data)

    table = pt.PrettyTable([config['dataset'], 'sentences', 'entities'])
    table.add_row(['train', len(train_data), train_ent_num])
    table.add_row(['test', len(test_data), test_ent_num])
    table.add_row(['dev', len(dev_data), dev_ent_num])
    config['logger'].info("\n{}".format(table))
    config['logger'].info(f"\n{vocab.label2id}")
    config['label_num'] = len(vocab.label2id)

    config['vocab'] = vocab
    config['pinyin_vocab'] = pinyin_vocab

    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab, char2img, pinyin_vocab))
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab, char2img, pinyin_vocab))
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab, char2img, pinyin_vocab))

    return (train_dataset, test_dataset, dev_dataset), (train_data, test_data, dev_data)
    # return (train_dataset, test_dataset), (train_data, test_data)
