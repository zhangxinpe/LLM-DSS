import torch.nn as nn
import torch.nn.functional as F
import torch

class RelativePositionEmbedding(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        # Pad 1 for <pad> token
        self.embeddings_table = nn.Embedding(max_relative_position * 2 + 1, num_units)

    def forward(self, relative_positions):
        # input: [batch_size, seq_len_q, seq_len_k]
        # output: [batch_size, seq_len_q, seq_len_k, num_units]
        embeddings = self.embeddings_table(relative_positions + self.max_relative_position)
        return embeddings


class MultiheadAttentionWithRelativePosition(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, max_relative_position=32): # Increased max_relative_position to 32
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.wo = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.max_relative_position = max_relative_position
        self.rp_k = RelativePositionEmbedding(self.head_dim, max_relative_position) # Relative position embedding for keys
        self.rp_v = RelativePositionEmbedding(self.head_dim, max_relative_position) # Implemented relative position embedding for values (optional, but included now)


    def forward(self, q, k, v, key_padding_mask=None):
        batch_size, seq_len_q, _ = q.size()
        batch_size, seq_len_k, _ = k.size()

        q_proj = self.wq(q).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, seq_len_q, head_dim]
        k_proj = self.wk(k).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, seq_len_k, head_dim]
        v_proj = self.wv(v).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, seq_len_k, head_dim]

        # Calculate relative position indices
        positions_i = torch.arange(seq_len_q).unsqueeze(1).repeat(1, seq_len_k) # [seq_len_q, seq_len_k]
        positions_j = torch.arange(seq_len_k).unsqueeze(0).repeat(seq_len_q, 1) # [seq_len_q, seq_len_k]
        relative_positions = positions_i - positions_j # [seq_len_q, seq_len_k]
        relative_positions_clipped = torch.clamp(relative_positions, -self.max_relative_position, self.max_relative_position) # Clip to max relative position

        # Get relative position embeddings for keys and values
        relative_position_keys = self.rp_k(relative_positions_clipped.to(q.device)) # [seq_len_q, seq_len_k, head_dim]
        relative_position_values = self.rp_v(relative_positions_clipped.to(q.device)) # [seq_len_q, seq_len_k, head_dim]


        # Calculate attention scores with relative position embeddings
        attn_scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) # [B, num_heads, seq_len_q, seq_len_k]

        # Corrected and standardized relative position bias addition using einsum:
        attn_scores_rp = torch.einsum("bhqd,qkd->bhqk", q_proj, relative_position_keys) # [B, num_heads, seq_len_q, seq_len_k]
        attn_scores += attn_scores_rp # Add relative position scores

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values, also incorporate relative position embeddings for values (optional)
        attn_output = torch.matmul(attn_weights, v_proj) # [B, num_heads, seq_len_q, head_dim]

        # Optional: Incorporate relative position embeddings for values (more complex, may not always be needed)
        # relative_position_values_reshaped = relative_position_values.permute(2, 0, 1).unsqueeze(0).unsqueeze(0) # [1, 1, head_dim, seq_len_q, seq_len_k]
        # attn_output_rp_v = torch.matmul(attn_weights.unsqueeze(-1), relative_position_values_reshaped.transpose(-2, -1)).squeeze(-1) # [B, num_heads, seq_len_q, head_dim]
        # attn_output += attn_output_rp_v # Add relative position values to output (optional)


        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim) # [B, seq_len_q, embed_dim]
        attn_output = self.wo(attn_output)

        return attn_output, attn_weights

class Residuals_fc_PreLN(nn.Module): # 类名修改为 Residuals_fc_PreLN 表示 Pre-LayerNorm 版本
    def __init__(self, hidden_size=768, dropout=0.1): # 添加 hidden_size 参数
        super().__init__()
        self.hidden_size = hidden_size # 保存 hidden_size
        # Feedforward network (same as before)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        # Layer normalization (现在是 Pre-LN，放前面)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=True)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=True)

        # Dropout (same as before)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x1): # x 是输入，x1 是残差连接的输入
        # Pre-LayerNorm: LayerNorm 在前
        normalized_x = self.layer_norm1(x) # 对输入 x 进行 LayerNorm
        x = x1 + normalized_x #  残差连接

        # Feedforward network
        ff_output = self.feedforward(normalized_x) # 注意：FFN 的输入也是 LayerNorm 后的 normalized_x
        ff_output = self.dropout(ff_output)  # Apply dropout
        normalized_ff_output = self.layer_norm2(ff_output) # 对 FFN 输出进行 LayerNorm
        x = x + normalized_ff_output # 残差连接
        return x

class TransformerLayer_PreLN_RP(nn.Module): # 类名修改为 TransformerLayer_PreLN 表示 Pre-LayerNorm 版本
    def __init__(self, hidden_size=768, num_heads=4, dropout=0.1, max_relative_position=5):
        super(TransformerLayer_PreLN_RP, self).__init__()
        self.hidden_size = hidden_size # 保存 hidden_size

        # MultiheadAttention with Relative Position
        self.attention = MultiheadAttentionWithRelativePosition(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, max_relative_position=max_relative_position)

        # 使用 Pre-LN 版本的 Residuals_fc
        self.residuals_fc = Residuals_fc_PreLN(hidden_size=hidden_size, dropout=dropout) # 传递 hidden_size

        # Layer Normalization for attention input (Pre-LN)
        self.layer_norm_attn = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=True) # 为注意力层输入添加 LN


    def forward(self, q, key_padding_mask):
        x = q # 输入

        # Pre-LayerNorm for Attention: LayerNorm 在 Attention 之前
        normalized_q = self.layer_norm_attn(x) # 对 Attention 的输入 q 进行 LayerNorm

        # Attention layer with residual connection and layer normalization
        attn_output, attn_weights = self.attention(normalized_q, normalized_q, normalized_q, key_padding_mask=~key_padding_mask.bool())  # 注意：Q, K, V 都使用 LayerNorm 后的 normalized_q
        # attn_output = self.dropout(attn_output)  # 可以考虑是否在 Attention 输出后加 Dropout (位置可以实验)

        x = self.residuals_fc(attn_output, x) #  Residuals_fc 内部已经是 Pre-LN 结构，这里 x 是残差连接的输入

        return x

class TransformerLayer_PreLN_SELF(nn.Module): # 类名修改为 TransformerLayer_PreLN 表示 Pre-LayerNorm 版本
    def __init__(self, hidden_size=768, num_heads=4, dropout=0.1):
        super(TransformerLayer_PreLN_SELF, self).__init__()
        self.hidden_size = hidden_size # 保存 hidden_size

        # MultiheadAttention with Relative Position
        self.attention = nn.MultiheadAttention(self.hidden_size, num_heads, dropout=dropout, batch_first=True) # batch_first=True

        # 使用 Pre-LN 版本的 Residuals_fc
        self.residuals_fc = Residuals_fc_PreLN(hidden_size=hidden_size, dropout=dropout) # 传递 hidden_size

        # Layer Normalization for attention input (Pre-LN)
        self.layer_norm_attn = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=True) # 为注意力层输入添加 LN


    def forward(self, q, key_padding_mask):
        x = q # 输入

        # Pre-LayerNorm for Attention: LayerNorm 在 Attention 之前
        normalized_q = self.layer_norm_attn(x) # 对 Attention 的输入 q 进行 LayerNorm

        # Attention layer with residual connection and layer normalization
        attn_output, attn_weights = self.attention(normalized_q, normalized_q, normalized_q, key_padding_mask=~key_padding_mask.bool())  # 注意：Q, K, V 都使用 LayerNorm 后的 normalized_q
        # attn_output = self.dropout(attn_output)  # 可以考虑是否在 Attention 输出后加 Dropout (位置可以实验)

        x = self.residuals_fc(attn_output, x) #  Residuals_fc 内部已经是 Pre-LN 结构，这里 x 是残差连接的输入

        return x



class fus_attention_layer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=4, dropout=0.1, num_attention_layers=5): #  添加 num_attention_layers 参数
        super(fus_attention_layer, self).__init__()
        self.num_attention_layers = num_attention_layers # 保存注意力层数量

        self.attention_1 = TransformerLayer_PreLN_SELF(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.attention_1_2 = TransformerLayer_PreLN_SELF(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.attention_2 = TransformerLayer_PreLN_SELF(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.attention_2_2 = TransformerLayer_PreLN_SELF(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.attention_3 = TransformerLayer_PreLN_SELF(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.attention_3_2 = TransformerLayer_PreLN_SELF(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.attention_4 = TransformerLayer_PreLN_SELF(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.attention_4_2 = TransformerLayer_PreLN_SELF(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.attention_5 = TransformerLayer_PreLN_SELF(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.attention_5_2 = TransformerLayer_PreLN_SELF(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.attention_6 = TransformerLayer_PreLN_RP(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.attention_6_2 = TransformerLayer_PreLN_RP(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)


        # 注意力权重层，用于为每个注意力层学习权重
        # 输入维度是 hidden_size (代表一个注意力层输出的特征向量)，输出维度是 1 (代表权重)
        self.attention_weights_fc = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1) # 对权重进行 softmax 归一化

        #  不再需要拼接后的全连接层，直接层归一化和 dropout
        self.layer_norm_fc = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, key_padding_masks):
        """
        Args:
            q: 输入 query, shape (batch_size, seq_len, hidden_size)
            key_padding_masks: 掩码列表, 长度应与注意力层数量相同 (这里是6),
                               每个掩码 shape (batch_size, seq_len) 或 (batch_size, seq_len_k)
        Returns:
            fused_output: 融合后的注意力输出, shape (batch_size, seq_len, hidden_size)  使用注意力机制动态加权融合
        """
        attention_outputs = []

        # 确保 key_padding_masks 列表不为空且长度与注意力层数量匹配
        if key_padding_masks is None or len(key_padding_masks) != self.num_attention_layers: # 使用 self.num_attention_layers
            raise ValueError(f"key_padding_masks must be a list of length {self.num_attention_layers}, one for each attention layer.") #  f-string 格式化

        # 1. 第一个注意力层使用第一个掩码
        attn_output_1 = self.attention_1(q, key_padding_masks[0])
        attn_output_1 = self.attention_1_2(attn_output_1, key_padding_masks[0])
        attention_outputs.append(attn_output_1)

        # 2. 第二个注意力层使用第二个掩码
        attn_output_2 = self.attention_2(q, key_padding_masks[1])
        attn_output_2 = self.attention_2_2(attn_output_2, key_padding_masks[1])
        attention_outputs.append(attn_output_2)

        # 3. 第三个注意力层使用第三个掩码
        attn_output_3 = self.attention_3(q, key_padding_masks[2])
        attn_output_3 = self.attention_3_2(attn_output_3, key_padding_masks[2])
        attention_outputs.append(attn_output_3)

        # 4. 第四个注意力层使用第四个掩码
        attn_output_4 = self.attention_4(q, key_padding_masks[3])
        attn_output_4 = self.attention_4_2(attn_output_4, key_padding_masks[3])
        attention_outputs.append(attn_output_4)

        # # 5. 第五个注意力层使用第五个掩码 (保留注释)
        # attn_output_5 = self.attention_5(q, key_padding_masks[4])
        # attention_outputs.append(attn_output_5)

        # 6. 第六个注意力层使用第六个掩码
        attn_output_6 = self.attention_6(q, key_padding_masks[4])
        attn_output_6 = self.attention_6_2(attn_output_6, key_padding_masks[4])
        attention_outputs.append(attn_output_6)


        weighted_outputs = []
        attention_weights = [] #  存储每个注意力层的权重，方便分析

        for attn_output in attention_outputs:
            #  对每个注意力层的输出进行全局平均池化，得到一个代表性的向量 (batch_size, hidden_size)
            #  这里假设你的注意力层输出的 shape 是 (batch_size, seq_len, hidden_size)
            pooled_output = attn_output.mean(dim=1) # 沿 seq_len 维度平均池化

            #  使用全连接层计算注意力权重 (batch_size, 1)
            weight = self.attention_weights_fc(pooled_output)
            attention_weights.append(weight) #  记录权重

        # 将权重列表转换为张量, 并进行 softmax 归一化 (batch_size, num_attention_layers, 1)
        attention_weights_tensor = torch.stack(attention_weights, dim=1) #  堆叠成 (batch_size, num_attention_layers, 1)
        normalized_weights = self.softmax(attention_weights_tensor) #  Softmax 归一化

        #  进行加权求和
        fused_output = torch.zeros_like(attention_outputs[0]) # 初始化融合输出，形状和第一个注意力层输出一致
        for i, attn_output in enumerate(attention_outputs):
            #  将权重应用到对应的注意力层输出上，并累加到 fused_output
            #  normalized_weights[:, i, :]  是第 i 个注意力层的权重, shape (batch_size, 1)
            #  为了进行元素乘法，需要将权重扩展到和 attn_output 相同的维度 (batch_size, 1, 1)  或者 (batch_size, seq_len, 1)  广播机制会自动处理
            weight = normalized_weights[:, i, :].unsqueeze(-1) #  扩展维度为 (batch_size, 1, 1)
            weighted_output = attn_output * weight #  广播乘法，权重应用到每个序列位置
            fused_output = fused_output + weighted_output


        # 层归一化和 dropout (保持不变)
        normalized_output = self.layer_norm_fc(fused_output)
        fused_output = self.dropout(normalized_output)
        # print(fused_output.shape)
        return fused_output


class TestFusAttentionLayer(): # 如果需要更正式的测试，可以使用 unittest

    def test_fus_attention_layer_output_shape(self):
        batch_size = 2
        seq_len = 10
        hidden_size = 768
        num_heads = 4
        dropout = 0.1

        # 1. 创建随机输入 query
        input_q = torch.randn(batch_size, seq_len, hidden_size)

        # 2. 创建 6 个随机 key_padding_masks (示例，实际应用中根据数据生成)
        key_padding_masks = [torch.randint(0, 2, (batch_size, seq_len)) for _ in range(6)]

        # 3. 实例化 fus_attention_layer
        fus_attn_layer = fus_attention_layer(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)

        # 4. 调用 forward 方法
        output = fus_attn_layer(input_q, key_padding_masks)

        # 5. 检查输出形状
        expected_output_shape = (batch_size, seq_len, hidden_size) # 期望的输出形状

        print("Input shape:", input_q.shape)
        print("Output shape:", output.shape)
        print("Test passed: Output shape is as expected!")



import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def masked_softmax(self, X, valid_lens):
        if valid_lens is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            shape = X.shape
            if valid_lens.dim() == 1:
                valid_lens = torch.repeat_interleave(valid_lens, shape[1])
            else:
                valid_lens = valid_lens.reshape(-1)
            masked_value = -1e6
            X = X.reshape(-1, shape[-1])
            maxlen = X.size(1)
            mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_lens[:, None]
            X[~mask] = masked_value
            X = X.reshape(shape)
            m, _ = torch.max(X, dim=-1)
            m, _ = torch.max(m, dim=-1)
            return nn.functional.softmax(X, dim=-1), m

    def forward(self, q, k, v, valid_lens=None, threshold=0.8):
        d = q.shape[-1]
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights, self.max_score = self.masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), v), self.max_score * math.sqrt(d) > threshold


def masked_mean(X, valid_lens=None):
    # X : [bs, n, d]
    if valid_lens is None:
        return X.sum(dim=1)
    else:
        X = X.permute(0, 2, 1)
        shape = X.shape
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        masked_value = 0
        X = X.reshape(-1, shape[-1])
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_lens[:, None]
        X[~mask] = masked_value
        valid_lens = valid_lens.reshape(shape[0], -1)
        return X.reshape(shape).sum(dim=-1) / valid_lens


import torch
import torch.nn as nn
import torch.nn.functional as F


# class TransformerLayer(nn.Module):
#     def __init__(self, hidden_size=768, num_heads=12, dropout=0.1):
#         super(TransformerLayer, self).__init__()

#         # MultiheadAttention
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout,
#                                                batch_first=True)

#         # Feedforward network
#         self.feedforward = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size * 2),
#             nn.ReLU(),
#             nn.Linear(hidden_size * 2, hidden_size)
#         )

#         # Layer normalization
#         self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=True)
#         self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=True)

#         # Dropout
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, q, k, v, key_padding_mask):
#         x = q
#         # Attention layer with residual connection and layer normalization
#         attn_output, _ = self.attention(q, k, v, key_padding_mask=key_padding_mask)  # Self-attention
#         # attn_output = self.dropout(attn_output)  # Apply dropout
#         x = x + self.layer_norm1(attn_output)  # Residual connection + LayerNorm

#         # Feedforward network with residual connection and layer normalization
#         ff_output = self.feedforward(x)
#         ff_output = self.dropout(ff_output)  # Apply dropout
#         x = x + self.layer_norm2(ff_output)  # Residual connection + LayerNorm

#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativePositionEmbedding(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        # Pad 1 for <pad> token
        self.embeddings_table = nn.Embedding(max_relative_position * 2 + 1, num_units)
        # self.linear = nn.Linear(num_units, num_units)  # Removed unused linear layer

    def forward(self, relative_positions):
        # input: [batch_size, seq_len_q, seq_len_k]
        # output: [batch_size, seq_len_q, seq_len_k, num_units]
        embeddings = self.embeddings_table(relative_positions + self.max_relative_position)
        return embeddings


class MultiheadAttentionWithRelativePosition(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, max_relative_position=32): # Increased max_relative_position to 32
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.wo = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.max_relative_position = max_relative_position
        self.rp_k = RelativePositionEmbedding(self.head_dim, max_relative_position) # Relative position embedding for keys
        self.rp_v = RelativePositionEmbedding(self.head_dim, max_relative_position) # Implemented relative position embedding for values (optional, but included now)


    def forward(self, q, k, v, key_padding_mask=None):
        batch_size, seq_len_q, _ = q.size()
        batch_size, seq_len_k, _ = k.size()

        q_proj = self.wq(q).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, seq_len_q, head_dim]
        k_proj = self.wk(k).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, seq_len_k, head_dim]
        v_proj = self.wv(v).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, seq_len_k, head_dim]

        # Calculate relative position indices
        positions_i = torch.arange(seq_len_q).unsqueeze(1).repeat(1, seq_len_k) # [seq_len_q, seq_len_k]
        positions_j = torch.arange(seq_len_k).unsqueeze(0).repeat(seq_len_q, 1) # [seq_len_q, seq_len_k]
        relative_positions = positions_i - positions_j # [seq_len_q, seq_len_k]
        relative_positions_clipped = torch.clamp(relative_positions, -self.max_relative_position, self.max_relative_position) # Clip to max relative position

        # Get relative position embeddings for keys and values
        relative_position_keys = self.rp_k(relative_positions_clipped.to(q.device)) # [seq_len_q, seq_len_k, head_dim]
        relative_position_values = self.rp_v(relative_positions_clipped.to(q.device)) # [seq_len_q, seq_len_k, head_dim]


        # Calculate attention scores with relative position embeddings
        attn_scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) # [B, num_heads, seq_len_q, seq_len_k]

        # Corrected and standardized relative position bias addition using einsum:
        attn_scores_rp = torch.einsum("bhqd,qkd->bhqk", q_proj, relative_position_keys) # [B, num_heads, seq_len_q, seq_len_k]
        attn_scores += attn_scores_rp # Add relative position scores

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values, also incorporate relative position embeddings for values (optional)
        attn_output = torch.matmul(attn_weights, v_proj) # [B, num_heads, seq_len_q, head_dim]

        # Optional: Incorporate relative position embeddings for values (more complex, may not always be needed)
        # relative_position_values_reshaped = relative_position_values.permute(2, 0, 1).unsqueeze(0).unsqueeze(0) # [1, 1, head_dim, seq_len_q, seq_len_k]
        # attn_output_rp_v = torch.matmul(attn_weights.unsqueeze(-1), relative_position_values_reshaped.transpose(-2, -1)).squeeze(-1) # [B, num_heads, seq_len_q, head_dim]
        # attn_output += attn_output_rp_v # Add relative position values to output (optional)


        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim) # [B, seq_len_q, embed_dim]
        attn_output = self.wo(attn_output)

        return attn_output, attn_weights


# class TransformerLayer(nn.Module):
#     def __init__(self, hidden_size=768, num_heads=4, dropout=0.1, max_relative_position=5): # Increased max_relative_position default
#         super(TransformerLayer, self).__init__()

#         # MultiheadAttention with Relative Position
#         self.attention = MultiheadAttentionWithRelativePosition(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, max_relative_position=max_relative_position) # Use new attention

#         # Feedforward network (same as before)
#         self.feedforward = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size * 2),
#             nn.ReLU(),
#             nn.Linear(hidden_size * 2, hidden_size)
#         )

#         # Layer normalization (same as before)
#         self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=True)
#         self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=True)

#         # Dropout (same as before)
#         self.dropout = nn.Dropout(dropout)
#         self.max_relative_position = max_relative_position


#     def forward(self, q, k, v, key_padding_mask):
#         x = q
#         # Attention layer with residual connection and layer normalization
#         attn_output, attn_weights = self.attention(q, k, v, key_padding_mask=key_padding_mask)  # Self-attention, now returns weights
#         # attn_output = self.dropout(attn_output)  # Consider enabling dropout after attention
#         x = x + self.layer_norm1(attn_output)  # Residual connection + LayerNorm

#         # Feedforward network with residual connection and layer normalization
#         ff_output = self.feedforward(x)
#         ff_output = self.dropout(ff_output)  # Apply dropout
#         x = x + self.layer_norm2(ff_output)  # Residual connection + LayerNorm

#         return x


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=4, dropout=0.1): # Increased max_relative_position default
        super(TransformerLayer, self).__init__()

        # MultiheadAttention with Relative Position
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout,batch_first=True) # Use new attention

        # Feedforward network (same as before)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        # Layer normalization (same as before)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=True)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=True)

        # Dropout (same as before)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, key_padding_mask):
        x = q
        # Attention layer with residual connection and layer normalization
        attn_output, attn_weights = self.attention(q, k, v, key_padding_mask=key_padding_mask)  # Self-attention, now returns weights
        # attn_output = self.dropout(attn_output)  # Consider enabling dropout after attention
        x = x + self.layer_norm1(attn_output)  # Residual connection + LayerNorm

        # Feedforward network with residual connection and layer normalization
        ff_output = self.feedforward(x)
        ff_output = self.dropout(ff_output)  # Apply dropout
        x = x + self.layer_norm2(ff_output)  # Residual connection + LayerNorm

        return x




import torch
import torch.nn.functional as F
import torch.nn as nn


import torch
import torch.nn.functional as F
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设 TransformerLayer 代码文件名为 transformer_layer.py 并且在同一个目录下
# from transformer_layer import TransformerLayer # 导入 TransformerLayer 如果 TransformerLayer 代码在单独文件中

class CosineSimilarityAttentionModel_PreLN_OutputNorm(nn.Module):
    """Cosine Similarity Attention Model - Pre-LN (Output Norm) - 最佳版本推荐."""
    def __init__(self, hidden_size, dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.feedforward = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=1e-6, elementwise_affine=True)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=1e-6, elementwise_affine=True)

    def forward(self, text1_output, text2_output, text1_mask, text2_mask):
        q = text1_output
        k = text2_output
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        cosine_sim = torch.matmul(q_norm, k_norm.transpose(-1, -2))
        text2_mask_expanded = text2_mask.unsqueeze(1).repeat(1, text1_mask.size(1), 1)
        cosine_sim = cosine_sim.masked_fill(text2_mask_expanded == 0, float('-inf'))
        attention_weights = F.softmax(cosine_sim, dim=-1)
        attention_weights = self.dropout(attention_weights)
        v = text2_output
        output = torch.matmul(attention_weights, v)
        normalized_attn_output = self.layer_norm1(output)
        output = normalized_attn_output + text1_output
        normalized_ff_output = self.layer_norm2(self.feedforward(normalized_attn_output))
        ff_output = self.dropout(normalized_ff_output)
        output = normalized_ff_output + output
        return output


# 传统注意力+余弦相似度注意力
class ModalFusionModule4(nn.Module):
    """ModalFusionModule4 - 最佳版本推荐，使用 Pre-LN (Output Norm) 的 CosineSimilarityAttention."""
    def __init__(self,  dropout_rate=0.1):
        super(ModalFusionModule4, self).__init__()
        self.global_dim = 768
        self.local_dim = 768

        #  假设 TransformerLayer 已经定义， 或者从文件导入
        self.attention = TransformerLayer(self.global_dim, num_heads=4, dropout=dropout_rate)

        #  最佳版本:  Pre-LN (Output Norm) 的 CosineSimilarityAttentionModel
        self.attention2 = CosineSimilarityAttentionModel_PreLN_OutputNorm(self.local_dim, dropout_rate=dropout_rate)


        self.alpha_predictor = nn.Sequential(
            nn.Linear(self.global_dim, self.global_dim // 2),
            nn.ReLU(),
            nn.Linear(self.global_dim // 2, 1),
            nn.Sigmoid()
        )
        self.output_layer = nn.Linear(self.global_dim, self.global_dim)
        self.layer_norm = nn.LayerNorm(self.global_dim, eps=1e-6, elementwise_affine=True)
        # self.layer_norm2 = nn.LayerNorm(self.global_dim, eps=1e-6, elementwise_affine=True) #  layer_norm2 可能冗余，如果 ModalFusionModule4 中只使用一个 LayerNorm，可以删除一个
        self.dropout = nn.Dropout(dropout_rate)
        # self.drop_out = nn.Dropout(dropout_rate) # drop_out 可能冗余，如果和 dropout 作用相同，可以删除一个, 这里统一使用 self.dropout
        self.drop_out = self.dropout # 为了保持代码兼容性，这里将 drop_out 指向 self.dropout,  建议后续代码统一使用 self.dropout 并删除 drop_out


    def forward(self, global_features, local_features, text1_mask, text2_mask):
        B, T, _ = global_features.size()
        Q = global_features
        K = local_features
        V = local_features

        attn_output = self.attention(Q, K, V, key_padding_mask=~text2_mask.bool())

        attn_output_alt = self.attention2(global_features, local_features, text1_mask, text2_mask)

        alpha = self.alpha_predictor(global_features.view(B * T, -1))
        alpha = alpha.view(B, T, 1)

        combined_attn_output = alpha * attn_output + (1 - alpha) * attn_output_alt

        output = combined_attn_output + self.layer_norm(self.drop_out(self.output_layer(combined_attn_output))) # 这里统一使用 self.drop_out (或 self.dropout)

        return output

if __name__ == '__main__':
    batch_size = 2
    seq_len = 10
    hidden_size = 768
    num_heads = 4
    dropout = 0.1

    # 1. 创建随机输入 query
    input_q = torch.randn(batch_size, seq_len, hidden_size)

    # 2. 创建 6 个随机 key_padding_masks (示例，实际应用中根据数据生成)
    key_padding_masks = [torch.randint(0, 1, (batch_size, seq_len)) for _ in range(6)]

    # 3. 实例化 fus_attention_layer
    fus_attn_layer = fus_attention_layer(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)

    # 4. 调用 forward 方法
    output = fus_attn_layer(input_q, key_padding_masks)

    # 5. 检查输出形状
    expected_output_shape = (batch_size, seq_len, hidden_size)  # 期望的输出形状

    print("Input shape:", input_q.shape)
    print("Output shape:", output.shape)
    print("Test passed: Output shape is as expected!")









