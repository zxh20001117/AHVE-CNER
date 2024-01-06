import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from Utils.MyDataset import RelationDataset, process_bert, collate_fn
from Utils.utils import LabelVocab

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
config = json.load(
    open("./config.json", 'r', encoding='utf-8')
)
with open(f"{config['root_path']}data/{config['dataset']}/train.json", 'r', encoding='utf-8') as f:
    train_data = json.load(f)
tokenizer = AutoTokenizer.from_pretrained(config['bert_path'])
vocab = LabelVocab()
train_ent_num = vocab.fill_vocab(train_data)
train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab))

train_loader = DataLoader(dataset=train_dataset,
                   batch_size=config['batch_size'],
                   collate_fn=collate_fn,
                   shuffle= False,
                   num_workers=0,
                   drop_last= False)

for i, data_batch in enumerate(train_loader):
    entity_text = data_batch[-1]
    data_batch = [data.cuda() for data in data_batch[:-1]]
    bert_inputs, word_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch
    print(entity_text)
    print(bert_inputs)
    print(word_inputs)
    print(grid_labels)
    print(grid_mask2d)
    print(pieces2word)
    print(dist_inputs)
    print(sent_length)
    break

