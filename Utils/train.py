import json

import numpy as np
import prettytable as pt
import torch
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score
from torch import nn

from Utils import utils


class Trainer(object):
    def __init__(self, model, config, updates_total, logger):
        self.model = model
        self.updates_total = updates_total
        self.logger = logger
        self.criterion = nn.CrossEntropyLoss()
        self.config = config
        self.device = config['device']
        bert_params = set(self.model.bert.parameters())
        pinyin_embedding_params = set(self.model.pinyin_encoder.pinyin_embedding.parameters())
        radical_embedding_params = set(self.model.radical_encoder.radical_embedder.char_embedding.parameters())
        # pinyin_embedding_params = set()
        # radical_embedding_params = set()
        embedding_params = list(pinyin_embedding_params.union(radical_embedding_params))
        other_params = list(set(self.model.parameters()) - bert_params - pinyin_embedding_params - radical_embedding_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.config['bert_learning_rate'],
             'weight_decay': self.config['weight_decay']},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.config['bert_learning_rate'],
             'weight_decay': 0.0},
            {'params': embedding_params,
             'lr': self.config['embedding_learning_rate'],
             'weight_decay': self.config['weight_decay']},
            {'params': other_params,
             'lr': self.config['learning_rate'],
             'weight_decay': self.config['weight_decay']},
        ]

        self.optimizer = transformers.AdamW(params, lr=self.config['learning_rate'],
                                            weight_decay=self.config['weight_decay'])
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=self.config[
                                                                                           'warm_factor'] * self.updates_total,
                                                                      num_training_steps=self.updates_total)

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []

        for i, data_batch in enumerate(data_loader):
            data_batch = [data.cuda() for data in data_batch[:-1]]

            bert_inputs, img_inputs, pinyin_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

            outputs = self.model(bert_inputs, img_inputs, pinyin_inputs, grid_mask2d, dist_inputs, pieces2word,
                                 sent_length)

            grid_mask2d = grid_mask2d.clone()
            loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad_norm'])
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            outputs = torch.argmax(outputs, -1)
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)

            label_result.append(grid_labels.cpu())
            pred_result.append(outputs.cpu())

            self.scheduler.step()

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [f1, p, r]])
        self.logger.info("\n{}".format(table))
        return f1

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()

        pred_result = []
        label_result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0

        confusion_matrix = utils.get_confusion_matrix(self.config['vocab'].id2label)
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, img_inputs, pinyin_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = self.model(bert_inputs, img_inputs, pinyin_inputs, grid_mask2d, dist_inputs, pieces2word,
                                 sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, _ = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy(),
                                                      confusion_matrix)

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "EVAL" if not is_test else "TEST"
        # self.logger.info('{} Label F1 {}'.format(title, f1_score(label_result.numpy(),
        #                                                          pred_result.numpy(),
        #                                                          average=None)))
        # self.logger.info('{} Label P {}'.format(title, precision_score(label_result.numpy(),
        #                                                                pred_result.numpy(),
        #                                                                average=None)))
        # self.logger.info('{} Label R {}'.format(title, recall_score(label_result.numpy(),
        #                                                             pred_result.numpy(),
        #                                                             average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])
        for k, v in confusion_matrix.items():
            t_f1, t_p, t_r = utils.cal_f1(v['c'], v['p'], v['r'])
            table.add_row(["<"+self.config['vocab'].id2label[k].upper()+">"] +
                          ["{:3.4f}".format(x) for x in [t_f1, t_p, t_r]])

        self.logger.info("\n{}".format(table))
        return e_f1
