import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel

from Modules.MyModel.GMU import GatedMultimodalLayer
from Modules.MyModel.PanGuTransformerEncoder import PanGuTransformerEncoder
from Modules.MyModel.PinyinEncoder import  CNNPinyinLevelEmbedding
from Modules.MyModel.RadicalEncoder import RadicalEncoder
from Modules.W2NER.CLN import CLN
from Modules.W2NER.CoPredictor import CoPredictor
from Modules.W2NER.MGDConv import ConvolutionLayer
from Utils.paths import bert_path


class Model(nn.Module):
    def __init__(self, config, tokenizer):
        super(Model, self).__init__()
        self.use_bert_last_4_layers = config['use_bert_last_4_layers']

        self.lstm_hid_size = config['lstm_hid_size']
        self.conv_hid_size = config['conv_hid_size']

        lstm_input_size = 0

        self.bert = AutoModel.from_pretrained(bert_path, cache_dir="./cache/", output_hidden_states=True)
        for param in self.bert.parameters():
            param.requires_grad_(False)

        lstm_input_size += config['bert_hid_size']

        self.dis_embs = nn.Embedding(20, config['dist_emb_size'])
        self.reg_embs = nn.Embedding(3, config['type_emb_size'])

        self.encoder = nn.LSTM(lstm_input_size, config['lstm_hid_size'] // 2, num_layers=1, batch_first=True,
                               bidirectional=True)

        conv_input_size = config['lstm_hid_size'] + config['dist_emb_size'] + config['type_emb_size']

        self.convLayer = ConvolutionLayer(conv_input_size, config['conv_hid_size'], config['dilation'],
                                          config['dropout']['conv_dropout'])
        self.dropout = nn.Dropout(config['dropout']['emb_dropout'])
        self.predictor = CoPredictor(config['label_num'], config['lstm_hid_size'], config['biaffine_size'],
                                     config['conv_hid_size'] * len(config['dilation']), config['ffnn_hid_size'],
                                     config['dropout']['out_dropout'])

        self.cln = CLN(config['lstm_hid_size'], config['lstm_hid_size'], conditional=True)

        self.patch_size = config['vit']['patch_size']
        self.patch_dim = config['vit']['patch_dim']
        self.feature_dim = config['bert_hid_size']
        self.radical_dim = config['vit']['patch_dim']
        self.dropouts = config['dropout']
        self.layer_preprocess_sequence = config['layer_preprocess_sequence']
        self.layer_postprocess_sequence = config['layer_postprocess_sequence']
        self.ff_size = config['ff_size']
        self.vit_max_num_token = config['vit']['max_num_token']
        self.vit_num_heads = config['vit']['num_heads']
        self.vit_num_layers = config['vit']['num_layers']
        self.radical_num_heads = config['radical']['num_heads']
        self.radical_num_layers = config['radical']['num_layers']
        self.font_size = config['vit']['font_size']

        self.pinyin_vocab = config['pinyin_vocab']
        self.radical_encoder = RadicalEncoder(patch_size=self.patch_size, patch_dim=self.patch_dim,
                                              radical_dim=self.radical_dim, feature_dim= self.feature_dim,
                                              dropout=self.dropouts,
                                              layer_preprocess_sequence=self.layer_preprocess_sequence,
                                              layer_postprocess_sequence=self.layer_postprocess_sequence,
                                              ff_size=self.ff_size,
                                              vit_max_num_token=self.vit_max_num_token,
                                              vit_num_heads=self.vit_num_heads,
                                              vit_num_layers=self.vit_num_layers,
                                              radical_num_heads=self.radical_num_heads,
                                              radical_num_layers=self.radical_num_layers, tokenizer=tokenizer)
        
        self.pinyin_encoder = CNNPinyinLevelEmbedding(self.pinyin_vocab, self.feature_dim, )
        
        self.pangu_encoder_pinyin = PanGuTransformerEncoder(emb_size=self.feature_dim)
        self.pangu_encoder_radical = PanGuTransformerEncoder(emb_size=self.feature_dim)
        self.GMU = GatedMultimodalLayer(emb_size=self.feature_dim)

    def forward(self, bert_inputs, img_inputs, pinyin_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length):
        """
        :param pinyin_inputs: [B, L, 3]
        :param img_inputs: [B, L, C, H, W]
        :param bert_inputs: [B, L']
        :param grid_mask2d: [B, L, L]
        :param dist_inputs: [B, L, L]
        :param pieces2word: [B, L, L']
        :param sent_length: [B]
        :return:
        """
        bert_embs = self.bert(input_ids=bert_inputs, attention_mask=bert_inputs.ne(0).float())
        if self.use_bert_last_4_layers:
            bert_embs = torch.stack(bert_embs[2][-4:], dim=-1).mean(-1)
        else:
            bert_embs = bert_embs[0]

        radical_embs = self.radical_encoder(bert_inputs, img_inputs)
        pinyin_embs = self.pinyin_encoder(pinyin_inputs)
        radical_embs = self.pangu_encoder_radical(bert_embs, radical_embs, radical_embs)
        pinyin_embs = self.pangu_encoder_pinyin(bert_embs, pinyin_embs, pinyin_embs)
        
        fusion_embs = self.GMU(bert_embs, radical_embs, pinyin_embs)

        length = pieces2word.size(1)

        min_value = torch.min(fusion_embs).item()

        # Max pooling word representations from pieces
        _fusion_embs = fusion_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _fusion_embs = torch.masked_fill(_fusion_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_fusion_embs, dim=2)

        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())

        cln = self.cln(word_reps.unsqueeze(2), word_reps)

        dis_emb = self.dis_embs(dist_inputs)
        tril_mask = torch.tril(grid_mask2d.clone().long())
        reg_inputs = tril_mask + grid_mask2d.clone().long()
        reg_emb = self.reg_embs(reg_inputs)

        conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)
        conv_inputs = torch.masked_fill(conv_inputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        conv_outputs = self.convLayer(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        outputs = self.predictor(word_reps, word_reps, conv_outputs)
        assert not torch.isnan(outputs).any()

        return outputs
