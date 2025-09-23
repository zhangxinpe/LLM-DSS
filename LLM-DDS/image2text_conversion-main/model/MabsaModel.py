import torch
import torch.nn as nn
import torch.nn.functional as F
from dns.resolver import query
from torch.nn.functional import dropout
from torch.utils.hipify.hipify_python import value
from transformers import RobertaModel, BertModel, AutoModel
from model.attention import *
from model.gate import *


class TextImageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bert = AutoModel.from_pretrained(self.cfg.model)  # bert model
        self.bert2 = AutoModel.from_pretrained(self.cfg.model)  # bert model
        self.model_fusion = ModalFusionModule4(dropout_rate=self.cfg.dropout)
        self.model_fusion_1 = ModalFusionModule4(dropout_rate=self.cfg.dropout)
        self.self_gate = EnhancedFilterModule(hidden_size=768)
        self.fus_gate = GatedMultimodalLayer(hidden_size=768)
        # self.model_fusion_2 = ModalFusionModule4(dropout_rate=self.cfg.dropout)
        # self.model_fusion_3 = ModalFusionModule4(dropout_rate=self.cfg.dropout)
        self.dropout = nn.Dropout(self.cfg.dropout)
        # self.layer1=nn.Linear(self.bert.config.hidden_size,self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, 3)  # 3 classes
        self.project = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.weight1 = nn.Parameter(torch.tensor(0.8))  # 可训练的权重1
        self.weight2 = nn.Parameter(torch.tensor(0.2))  # 可训练的权重1
        self._init_weights(self.out)
        self._init_weights(self.project)
        self._init_weights(self.model_fusion)
        self._init_weights(self.self_gate)
        self._init_weights(self.fus_gate)
        self._init_weights(self.model_fusion_1)

    def _init_weights(self, module):
        if isinstance(module, nn.AdaptiveAvgPool1d):
            torch.nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, input_ids2, attention_mask2):

        # inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)  #[batch,128,768]

        # # Prediction Module
        outputs_text = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        outputs_text2 = self.bert2(input_ids=input_ids2, attention_mask=attention_mask2)

        tweet_cls = outputs_text.pooler_output

        tweet = outputs_text.last_hidden_state
        image = outputs_text2.last_hidden_state

        outputs = self.model_fusion(tweet, image, attention_mask, attention_mask2)

        outputs1 = self.model_fusion_1(outputs, image, attention_mask, attention_mask2)

        # outputs2 = self.model_fusion_2(tweet,caption,attention_mask,attention_mask2)

        cls_output = self.dropout(F.relu(self.project(outputs1)))

        cls_output =  torch.max(cls_output , dim=1).values

        cls_output = self.self_gate(cls_output)

        cls_output =  (cls_output + tweet_cls)/2

        cls_output = self.weight2*cls_output + self.weight1*tweet_cls



        # cls_output = self.fus_gate(cls_output, tweet_cls)

        outputs = self.out(cls_output)

        return outputs, cls_output

