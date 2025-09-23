import torch
import torch.nn as nn



        

class MeanPooling_batch(nn.Module):
    def __init__(self):
        super(MeanPooling_batch, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        """
        :param last_hidden_state: Tensor of shape (batch_size, num_seq, seq_len, dim)
        :param attention_mask: Tensor of shape (batch_size, num_seq, seq_len) with 1 for real tokens and 0 for padding tokens
        :return: mean_embeddings of shape (batch_size, num_seq, dim)
        """
        try:

            batch_size, num_seq, seq_len, dim = last_hidden_state.shape
        except:
            print(last_hidden_state.shape)
        
        # 扩展 attention_mask 到 (batch_size, num_seq, seq_len, dim)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(-1, -1, -1,last_hidden_state.size(3)).float()

        # 对有效的 token 的嵌入求和
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=2)

        # 计算有效 token 数量
        sum_mask = input_mask_expanded.sum(dim=2)
        
        # 防止除以 0
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # 求平均池化
        mean_embeddings = sum_embeddings / sum_mask

        return mean_embeddings
# # 测试函数
# mean_pooling_batch = MeanPooling_batch()
# inputs_emb_batch.shape
# attention_mask_batch.shape

class MeanPooling_last(nn.Module):
    def __init__(self):
        super(MeanPooling_last, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        """
        对 `last_hidden_state` 使用 `attention_mask` 进行 mean pooling。

        :param last_hidden_state: Tensor of shape (batch_size, seq_len, hidden_size)
        :param attention_mask: Tensor of shape (batch_size, seq_len), 1 表示有效 token，0 表示 padding
        :return: Tensor of shape (batch_size, hidden_size), mean pooled embeddings
        """
        # 扩展 attention_mask 到 (batch_size, seq_len, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(-1, -1, last_hidden_state.size(2)).float()  # shape: (batch_size, seq_len, hidden_size)

        # 按元素相乘，mask 将使 padding 部分为 0
        masked_embeddings = last_hidden_state * input_mask_expanded  # shape: (batch_size, seq_len, hidden_size)

        # 对有效 token 的嵌入求和
        sum_embeddings = masked_embeddings.sum(1)  # shape: (batch_size, hidden_size)

        # 计算有效 token 数量
        sum_mask = input_mask_expanded.sum(1)  # shape: (batch_size, hidden_size)

        # 防止除以 0
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        # 求平均池化
        mean_embeddings = sum_embeddings / sum_mask  # shape: (batch_size, hidden_size)

        return mean_embeddings
# 测试函数
# last_hidden_state = torch.randn(8, 12, 768)  # 假设输入的 last_hidden_state
# attention_mask = torch.randint(0, 2, (8, 12))  # 假设输入的 attention_mask
# mean_pooling_last=MeanPooling_last()

# mean_embeddings = mean_pooling_last(last_hidden_state, attention_mask)
# print(mean_embeddings.shape)  # 输出 (batch_size, hidden_size) -> (8, 768)

import torch
import torch.nn.functional as F

class SoftmaxFusion(nn.Module):
    def __init__(self,dropout,feature_dim):
        super(SoftmaxFusion, self).__init__()
        self.mean_pooling_batch = MeanPooling_batch()
        self.mean_pooling_last = MeanPooling_last()
        self.dropout = nn.Dropout(dropout)
        self.feedforward = nn.Linear(feature_dim, feature_dim)
        self.feedforward_1 = nn.Linear(feature_dim, feature_dim)
        self.feedforward_2 = nn.Linear(feature_dim, feature_dim)
        

    def forward(self, inputs_emb_aspect,attention_mask_aspect,inputs_emb_batch,attention_mask_batch):
        """
        :param seq_feats: Tensor of shape (batch_size, num_seq, dim)
        :param global_feat: Tensor of shape (batch_size, dim)
        :return: fused_feat: Tensor of shape (batch_size, dim)
        """

        local_feature_pooler = self.mean_pooling_batch(inputs_emb_batch, attention_mask_batch)
        local_feature_pooler = F.tanh(self.feedforward_1(local_feature_pooler))
        local_feature_pooler = self.dropout(local_feature_pooler)

        
        aspect_feature_pooler=self.mean_pooling_last(inputs_emb_aspect,attention_mask_aspect)
        aspect_feature_pooler = F.tanh(self.feedforward_2(aspect_feature_pooler))
        aspect_feature_pooler = self.dropout(aspect_feature_pooler)
        

        seq_feats = local_feature_pooler
        global_feat = aspect_feature_pooler

        # 计算每个序列和全局特征的相似度，使用点积
        # (batch_size, num_seq, dim) . (batch_size, dim) -> (batch_size, num_seq)
        sim_scores = torch.bmm(seq_feats, global_feat.unsqueeze(-1)).squeeze(-1)  # (batch_size, num_seq)

        # 使用 softmax 计算权重
        attention_weights = self.dropout(F.softmax(sim_scores, dim=-1))  # (batch_size, num_seq)

        # 1. 扩展权重张量，shape 从 (batch_size, 3) 扩展到 (batch_size, 3, 1, 1)
        weights_expanded = attention_weights.unsqueeze(-1).unsqueeze(-1)  # shape: (batch_size, 3, 1, 1)

        # 2. 广播权重到与向量形状一致
        weights_expanded = weights_expanded.expand(-1, -1, inputs_emb_batch.size(2), inputs_emb_batch.size(3))  # shape: (batch_size, 3, seq_len, dim)

        # 3. 按元素相乘
        weighted_vectors = inputs_emb_batch * weights_expanded  # shape: (batch_size, 3, seq_len, dim)

        final_result = weighted_vectors.sum(dim=1) # shape: (batch_size, seq_len, dim)

        final_result = F.tanh(self.feedforward(final_result)) # shape: (batch_size, seq_len, dim)
        final_result = self.dropout(final_result) # shape: (batch_size, seq_len, dim)

        maxlen_attention_mask , _ =attention_mask_batch.sum(dim=-1).max(dim=-1) # shape: (batch_size, seq_len)

        

        return final_result,maxlen_attention_mask

if __name__ == '__main__':
    inputs_emb_aspect=torch.randn(8,12,768)
    attention_mask_aspect=torch.randint(0,10,(8,12))
    # attention_mask_aspect.shape
    inputs_emb_batch=torch.randn(8,3,12,768)
    attention_mask_batch=torch.randint(0,10,(8,3,12))
    # attention_mask_batch.shape
    softmax_fusion=SoftmaxFusion(dropout=0.1,feature_dim=768)
    result,maxlen_attention_mask=softmax_fusion(inputs_emb_aspect,attention_mask_aspect,inputs_emb_batch,attention_mask_batch)
    print(result.shape)
    print(maxlen_attention_mask.shape)
    print(maxlen_attention_mask)
    pass
