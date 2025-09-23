import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedFilterModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate = self.gate(x)
        out = gate * x
        return out

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate):
        super().__init__()

        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.linear_2(self.relu(self.linear_1(x))))

class ClipModel(nn.Module):
    def __init__(self,args,clip_base_model ):
        super(ClipModel, self).__init__()

        # 输入特征维度
        self.image_feature_dim = args.clip_image_feature_dim
        self.text_feature_dim = args.clip_text_feature_dim

        self.clip_base_model=clip_base_model

        self.fusion_dim=512

        self.text_layer=torch.nn.Linear(self.text_feature_dim,self.fusion_dim)
        self.aspect_Layer=torch.nn.Linear(self.text_feature_dim,self.fusion_dim)
        self.image_Layer=torch.nn.Linear(self.image_feature_dim,self.fusion_dim)


        # 融合层：将所有特征连接在一起
        self.fc = nn.Linear(
            self.fusion_dim * 3,  # 加上hidden states
            self.fusion_dim
        )
        self.mlp=MLP(self.fusion_dim,self.fusion_dim,3,0.2)

        # 输出层


    def forward(self,clip_image, clip_text,clip_aspect):
        image_result = self.clip_base_model.encode_image(clip_image, output_hidden_states=True)  # 257 维度 第一个是cls
        text_result = self.clip_base_model.encode_text(clip_text, output_hidden_states=True)
        aspect_result=self.clip_base_model.encode_text(clip_aspect,output_hidden_states=True)
        image_features, image_hidden_states = image_result
        text_features, text_hidden_states = text_result
        aspect_features,aspect_hidden_states=aspect_result


        # image_features=F.relu(self.image_Layer(image_features))
        # text_features=F.relu(self.text_layer(text_features))
        # aspect_features=F.relu(self.aspect_Layer(aspect_features))


        # 特征融合：连接图像、文本、方面特征和隐藏状态
        fused_features = torch.cat((
            image_features,
            text_features,
            aspect_features,
        ), dim=-1)

        # 通过融合层
        fc_outputs = self.fc(fused_features)
        mlp_outputs = self.mlp(fc_outputs) # [batch,3]



        return mlp_outputs