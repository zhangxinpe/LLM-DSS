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
    def __init__(self, hidden_size=768, num_heads=4, dropout=0.1,key_padding_masks=None):
        super(fus_attention_layer, self).__init__()
        # len_attention=len(key_padding_masks) #  虽然这里计算了长度，但目前没有直接使用
        self.attention_1 = TransformerLayer_PreLN_SELF(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.attention_2 = TransformerLayer_PreLN_SELF(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.attention_3 = TransformerLayer_PreLN_SELF(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.attention_4 = TransformerLayer_PreLN_SELF(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.attention_5 = TransformerLayer_PreLN_SELF(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.attention_6 = TransformerLayer_PreLN_RP(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)

        # 全连接层，输入维度是拼接后的维度 (hidden_size * 6)，输出维度是 hidden_size
        self.fc = nn.Linear(hidden_size * 6, hidden_size)

        # 添加 Layer Normalization 层
        self.layer_norm_fc = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, key_padding_masks):
        """
        Args:
            q: 输入 query, shape (batch_size, seq_len, hidden_size)
            key_padding_masks: 掩码列表, 长度应与注意力层数量相同 (这里是6),
                               每个掩码 shape (batch_size, seq_len) 或 (batch_size, seq_len_k)
        Returns:
            fused_output: 融合后的注意力输出, shape (batch_size, seq_len, hidden_size)  使用拼接 + 全连接 + 层归一化融合
        """
        attention_outputs = []

        # 确保 key_padding_masks 列表不为空且长度与注意力层数量匹配
        if key_padding_masks is None or len(key_padding_masks) != 6:
            raise ValueError("key_padding_masks must be a list of length 6, one for each attention layer.")

        # 1. 第一个注意力层使用第一个掩码
        attn_output_1 = self.attention_1(q, key_padding_masks[0])
        attention_outputs.append(attn_output_1)

        # 2. 第二个注意力层使用第二个掩码
        attn_output_2 = self.attention_2(q, key_padding_masks[1])
        attention_outputs.append(attn_output_2)

        # 3. 第三个注意力层使用第三个掩码
        attn_output_3 = self.attention_3(q, key_padding_masks[2])
        attention_outputs.append(attn_output_3)

        # 4. 第四个注意力层使用第四个掩码
        attn_output_4 = self.attention_4(q, key_padding_masks[3])
        attention_outputs.append(attn_output_4)

        # 5. 第五个注意力层使用第五个掩码
        attn_output_5 = self.attention_5(q, key_padding_masks[4])
        attention_outputs.append(attn_output_5)

        # 6. 第六个注意力层使用第六个掩码
        attn_output_6 = self.attention_6(q, key_padding_masks[5])
        attention_outputs.append(attn_output_6)

        # 拼接所有注意力输出 (使用拼接融合)
        concatenated_output = torch.cat(attention_outputs, dim=-1)  # 沿最后一个维度 (hidden_size 维度) 拼接

        # 全连接层降维
        fc_output = self.fc(concatenated_output)

        # 层归一化
        normalized_output = self.layer_norm_fc(fc_output)

        fused_output = self.dropout(normalized_output) # Dropout 在 LayerNorm 之后

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









