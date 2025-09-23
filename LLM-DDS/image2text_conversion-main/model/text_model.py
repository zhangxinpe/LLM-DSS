
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, BertModel,AutoModel
from model.attention import *
from model.gate import *
# Construct the joint Caption & Sentiment Classifier Model
class OnlyTextModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bert = RobertaModel.from_pretrained(self.cfg.model) # bert model
        # self.bert = BertModel.from_pretrained(self.cfg.model) # bert model
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, 3) # 3 classes
        self.project = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        # self.attention = TransformerLayer(self.bert.config.hidden_size, num_heads=2, dropout=cfg.dropout,max_relative_position=4)
        self.fus_gate = GatedMultimodalLayer(hidden_size=768)
        self.self_gate = EnhancedFilterModule(hidden_size=768)
        self.fus_attention = fus_attention_layer(hidden_size=768, num_heads=4, dropout=self.cfg.dropout)
        self.fus_attention_1 = fus_attention_layer(hidden_size=768, num_heads=4, dropout=self.cfg.dropout)
        self.weight1 = nn.Parameter(torch.tensor(0.8))  # 可训练的权重1
        self.weight2 = nn.Parameter(torch.tensor(0.2))  # 可训练的权重1
        self._init_weights(self.out)
        self._init_weights(self.project)
        self._init_weights(self.fus_gate)
        # self._init_weights(self.attention )
        self._init_weights(self.self_gate)
        self._init_weights(self.fus_attention)
        self._init_weights(self.fus_attention_1)


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

    def forward(self,input_ids,attention_mask,attention_mask_fus):
   
        # Prediction Module
        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)

        pooler_output = outputs.pooler_output

        hidden_state = outputs.last_hidden_state


        attn_output = self.fus_attention(hidden_state, key_padding_masks=attention_mask_fus)

        # attn_output = self.fus_attention_1(attn_output, key_padding_masks=attention_mask_fus)


        attn_output = self.dropout(F.relu(self.project(attn_output)))

        # 最大池化
        attn_output =  torch.max(attn_output, dim=1).values
        # attn_output =  attn_output[:,0,:]


        attn_output = self.self_gate(attn_output)

        attn_output = (attn_output + pooler_output)/2

        # outputs = outputs.pooler_output

        outputs = self.weight2*attn_output + self.weight2*pooler_output



        outputs = self.out(self.dropout(outputs))


        return outputs,pooler_output
