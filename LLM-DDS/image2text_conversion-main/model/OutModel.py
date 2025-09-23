
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class StackedFusionModel_2(nn.Module):

    def __init__(self):
        super(StackedFusionModel_2, self).__init__()

        self.num_classes = 3
        initial_weights = torch.tensor([[0.2, 0.8],  # 类别 0: 文本权重 0.3, 多模态权重 0.7
                                        [0.2, 0.8],  # 类别 1: 文本权重 0.3, 多模态权重 0.7
                                        [0.2, 0.8]],  # 类别 2: 文本权重 0.3, 多模态权重 0.7
                                       dtype=torch.float32)

        # 将初始权重张量注册为可学习的参数
        self.class_weights = nn.Parameter(initial_weights)

        self.class_weights.requires_grad = False  # 阻止更新 class_weights

        # Activation functions
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        """
        Args:
            x1: Tensor of shape [batch_size, num_classes] (logits from model 1)
            x2: Tensor of shape [batch_size, num_classes] (logits from model 2)
        Returns:
            fus_output: Tensor of shape [batch_size, num_classes] (fused logits)
            class_weights_numpy: NumPy array of shape [num_classes, 2] (detached class weights)
        """
        prob1 = self.softmax(x1)
        prob2 = self.softmax(x2)

        batch_size = x1.size(0)

        # 扩展类别权重以匹配批次大小
        expanded_class_weights = self.class_weights.unsqueeze(0).repeat(batch_size, 1, 1) # [batch, num_classes, 2]

        # 扩展概率以进行逐元素相乘
        prob1_expanded = prob1.unsqueeze(-1) # [batch, num_classes, 1]
        prob2_expanded = prob2.unsqueeze(-1) # [batch, num_classes, 1]

        # 堆叠概率
        probs_stacked = torch.cat([prob1_expanded, prob2_expanded], dim=-1) # [batch, num_classes, 2]

        # 进行类别特定的加权平均
        fused_probs = torch.sum(expanded_class_weights * probs_stacked, dim=-1) # [batch, num_classes]

        fus_output = torch.log(fused_probs + 1e-6)

        class_weights_numpy = self.class_weights.detach().cpu().numpy()

        text_weights = class_weights_numpy[:, 0]  # 获取所有类别的文本权重
        average_text_weight = np.mean(text_weights)  # 计算平均值

        return fus_output, average_text_weight # 返回模型1的权重作为示例

class FusedModel_loss_1_1(nn.Module):

    def __init__(self, only_text_model,text_image_model):
        super().__init__()
        self.only_text_model=only_text_model
        self.text_image_model=text_image_model
        self.stack_fus = StackedFusionModel_2()


    def forward(self,input_ids, attention_mask,
                        input_ids_all, attention_mask_all,
                        input_ids_images, attention_mask_images,attention_mask_fus
                        ):

        text_image_output,text_image_pooler=self.text_image_model(
                        input_ids_all,attention_mask_all,
                        input_ids_images,attention_mask_images
                        )
        only_text_output,only_text_pooler=self.only_text_model(input_ids, attention_mask,attention_mask_fus)
        fus_output,weight=self.stack_fus(only_text_output,text_image_output)
        # fus_output,weight=self.stack_fus(only_text_output,text_image_output)
        return  only_text_output,text_image_output,fus_output,weight,1-weight


