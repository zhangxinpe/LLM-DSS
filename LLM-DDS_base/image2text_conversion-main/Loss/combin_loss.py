import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class kl_divergence(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p, q):
        p = F.softmax(p, dim=-1)
        q_log = F.log_softmax(q,dim=-1)

        # 使用 PyTorch 的 kl_div 函数计算 KL 散度
        # p 是概率分布，q_log 是 log(q)，即 KL(p || q)
        loss = F.kl_div(q_log, p, reduction='batchmean')  # reduction='batchmean' 计算批量平均损失
        return loss


class CombinedLoss_3_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = kl_divergence()
        self.NLLLoss = torch.nn.NLLLoss()

    def forward(self, only_text_output, text_image_output, fus_output, labels):
        kl_loss = self.kl_loss(text_image_output, only_text_output)
        
        all_loss = self.NLLLoss(fus_output,labels)

        # if kl_loss<=0.000001 or all_loss <= 0.000001:
        #     print(kl_loss, all_loss)
        #     print(fus_output)
        # print(fus_output[0],labels[0])
        total_loss = kl_loss + all_loss

        if total_loss<=0.0001:
            print(kl_loss, all_loss)
            print(fus_output)
        # kl_loss = self.kl_loss(text_image_output,only_text_output)
        # t_loss = self.ce_loss(only_text_output, labels)
        # v_loss = self.ce_loss(text_image_output, labels)


    # 计算 SoftHGRLoss
        # total_loss = kl_loss + t_loss  + v_loss

        return total_loss



class CombinedLoss_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss=kl_divergence()


    def forward(self, only_text_output, text_image_output, fus_output,labels):
            kl_loss = self.kl_loss(text_image_output,only_text_output)
            t_loss = self.ce_loss(only_text_output, labels)
            v_loss = self.ce_loss(text_image_output, labels)


        # 计算 SoftHGRLoss
            total_loss = kl_loss + t_loss  + v_loss
            # total_loss = t_loss + v_loss

            return total_loss




class CombinedLoss_4(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bt_loss=BarlowTwinsLoss()


    def forward(self, only_text_output, text_image_output, only_text_pooler,text_image_pooler,labels):
            bt_loss = self.bt_loss(text_image_pooler,only_text_pooler)
            t_loss = self.ce_loss(only_text_output, labels)
            v_loss = self.ce_loss(text_image_output, labels)

        # 计算 SoftHGRLoss
            total_loss = bt_loss + t_loss  + v_loss
            # total_loss = t_loss + v_loss

            return total_loss


class MutualDistillationModel(nn.Module):
    def __init__(self, T=2.0, alpha=0.5, beta=0.5):
        """
        参数:
            model_A: 模型A（例如文本模型），输入为文本
            model_B: 模型B（例如文本+图像模型），输入为文本和图像
            T: 温度参数，通常 T > 1
            alpha: 硬标签损失权重
            beta: 软标签损失权重
        """
        super(MutualDistillationModel, self).__init__()
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits_A,logits_B, labels):
        """
        参数:
            input_text: 文本输入，供模型A和模型B使用
            input_image: 图像输入，仅供模型B使用
            labels: 真实标签
        返回:
            total_loss: 两个模型总损失
            loss_A: 模型A的损失
            loss_B: 模型B的损失
        """

        # 1. 计算硬标签损失（交叉熵损失）
        loss_hard_A = F.cross_entropy(logits_A, labels)
        loss_hard_B = F.cross_entropy(logits_B, labels)

        # 2. 计算温度平滑后的概率分布（软标签）
        T = self.T
        p_A_T = F.softmax(logits_A / T, dim=1)
        p_B_T = F.softmax(logits_B / T, dim=1)

        # 3. 计算软标签损失（KL散度），注意输入要求为对数概率
        loss_soft_A = self.kl_loss_fn(F.log_softmax(logits_A / T, dim=1), p_B_T)
        loss_soft_B = self.kl_loss_fn(F.log_softmax(logits_B / T, dim=1), p_A_T)

        # 4. 计算每个模型的总损失（硬损失 + 温度平方因子的软损失）
        loss_A = self.alpha * loss_hard_A + self.beta * (T ** 2) * loss_soft_A
        loss_B = self.alpha * loss_hard_B + self.beta * (T ** 2) * loss_soft_B

        total_loss = loss_A + loss_B

        return total_loss

if __name__ == '__main__':
    # 假设的输入特征和标签
    f_t = torch.rand(32, 768)  # 文本特征 (32 个样本, 768 维)
    f_v = torch.rand(32, 768)  # 视觉特征 (32 个样本, 768 维)
    labels = torch.randint(0, 10, (32,))  # 假设有 10 类, 32 个样本的标签

    # 初始化损失函数
    loss_fn = MutualDistillationModel()


    # 计算损失
    loss = loss_fn(f_t, f_v,labels)

    # 打印损失
    print(loss)


