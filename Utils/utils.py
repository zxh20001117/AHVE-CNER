import logging
import time
from collections import defaultdict, deque

import numpy as np
from fastNLP import Vocabulary, DataSet
from pypinyin import pinyin, Style
from pypinyin.contrib.tone_convert import to_initials, to_finals
from pypinyin_dict.phrase_pinyin_data import cc_cedict
from transformers import AutoTokenizer

from Utils.paths import root_path, radical_vocab_path, pinyin_vocab_path


class LabelVocab(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]

    def fill_vocab(self, dataset):
        ent_num = 0
        for item in dataset:
            for ent in item['ner']:
                self.add_label(ent['type'])
                ent_num += 1
        return ent_num


class RadicalVocab():
    def __init__(self, tokenizer: AutoTokenizer, include_word_start_end=False):
        self.radical_path = root_path + radical_vocab_path
        self.char_tokenizer = tokenizer
        self.char_info = dict()

        with open(self.radical_path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                char, info = line.split('\t', 1)
                self.char_info[char] = info.replace('\n', '').split('\t')[0].split()

        self.char_info[tokenizer.pad_token] = ['<pad>']
        self.char_info[tokenizer.unk_token] = ['<unk>']
        self.char_info[tokenizer.cls_token] = ['<CLS>']
        self.char_info[tokenizer.sep_token] = ['<SEP>']

        self.radical_vocab = self.construct_radical_vocab_from_vocab(include_word_start_end=include_word_start_end)
        self.max_radical_nums = max(map(lambda x: len(self.char2radical(x)), self.char_tokenizer.vocab.keys()))

    def char2radical(self, c):
        if c in self.char_info.keys():
            c_info = self.char_info[c]
            return c_info
        return ['○']

    def construct_radical_vocab_from_vocab(self, min_freq: int = 1, include_word_start_end=False):
        r"""
        给定一个char的vocabulary生成character的vocabulary.
        :param min_freq:
        :param include_word_start_end: 是否需要包含特殊的<bow>和<eos>
        :return:
        """
        radical_vocab = Vocabulary(min_freq=min_freq)
        for char, index in self.char_tokenizer.vocab.items():
            radical_vocab.add_word_lst(self.char2radical(char))
        if include_word_start_end:
            radical_vocab.add_word_lst(['[CLS]', '[SEP]'])
        return radical_vocab


def pinyin_split(word):
    pingyin = pinyin(word, style=Style.TONE3, heteronym=True, strict=False)[0][0]
    initial = to_initials(pingyin, strict=False)
    final = to_finals(pingyin)
    initial = initial if initial != '' else '-'  # 有些字没有声母，比如“啊”， 或者 无法被分解为拼音
    final = final if final != '' else '-'  # 无法被分解为拼音
    yindiao = pingyin[-1] if pingyin[-1].isnumeric() else '-'
    return [initial, final, yindiao]


class PinyinVocab:
    def __init__(self, tokenizer: AutoTokenizer):
        cc_cedict.load()
        self.char_tokenizer = tokenizer
        self.pinyin_vocab_path = root_path + pinyin_vocab_path
        self.vocab = self.load_pinyin_vocab()

    def load_pinyin_vocab(self, min_freq: int = 1, include_word_start_end=False) -> Vocabulary:
        vocab = Vocabulary(min_freq=min_freq)
        vocab_dict = np.load(self.pinyin_vocab_path, allow_pickle=True).item()
        vocab.from_dataset(DataSet(vocab_dict), field_name='yinjie')
        vocab.add_word_lst(
            ['-', self.char_tokenizer.pad_token, self.char_tokenizer.unk_token, self.char_tokenizer.cls_token,
             self.char_tokenizer.sep_token])
        return vocab


def get_logger(dataset):
    pathname = "./log/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)


def decode(outputs, entities, length):
    class Node:
        def __init__(self):
            self.THW = []  # [(tail, type)]
            self.NNW = defaultdict(set)  # {(head,tail): {next_index}}

    ent_r, ent_p, ent_c = 0, 0, 0
    decode_entities = []
    q = deque()
    for instance, ent_set, l in zip(outputs, entities, length):
        predicts = []
        nodes = [Node() for _ in range(l)]
        for cur in reversed(range(l)):
            heads = []
            for pre in range(cur + 1):
                # THW
                if instance[cur, pre] > 1:
                    nodes[pre].THW.append((cur, instance[cur, pre]))
                    heads.append(pre)
                # NNW
                if pre < cur and instance[pre, cur] == 1:
                    # cur node
                    for head in heads:
                        nodes[pre].NNW[(head, cur)].add(cur)
                    # post nodes
                    for head, tail in nodes[cur].NNW.keys():
                        if tail >= cur and head <= pre:
                            nodes[pre].NNW[(head, tail)].add(cur)
            # entity
            for tail, type_id in nodes[cur].THW:
                if cur == tail:
                    predicts.append(([cur], type_id))
                    continue
                q.clear()
                q.append([cur])
                while len(q) > 0:
                    chains = q.pop()
                    for idx in nodes[chains[-1]].NNW[(cur, tail)]:
                        if idx == tail:
                            predicts.append((chains + [idx], type_id))
                        else:
                            q.append(chains + [idx])

        predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
        decode_entities.append([convert_text_to_index(x) for x in predicts])
        ent_r += len(ent_set)
        ent_p += len(predicts)
        ent_c += len(predicts.intersection(ent_set))
    return ent_c, ent_p, ent_r, decode_entities


def cal_f1(c, p, r):
    if r == 0 or p == 0:
        return 0, 0, 0

    r = c / r if r else 0
    p = c / p if p else 0

    if r and p:
        return 2 * p * r / (p + r), p, r
    return 0, p, r

