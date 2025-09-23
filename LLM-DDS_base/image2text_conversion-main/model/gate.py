from torch import nn
import torch
# torch.sigmoid()更适合快速计算;nn.Sigmoid()更适合构建可训练神经网络
# nn.Sigmoid()是一个nn.Module,可以作为神经网络模块使用,具有可学习的参数,可以通过反向传播训练。torch.sigmoid()是一个固定的数学函数。
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

class GatedMultimodalLayer(nn.Module):
    """
    Gated Multimodal Layer based on 'Gated multimodal networks,
    Arevalo1 et al.' (https://arxiv.org/abs/1702.01992)
    """

    def __init__(self, hidden_size):
        super(GatedMultimodalLayer, self).__init__()
        hidden_size = hidden_size

        self.hidden1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_sigmoid = nn.Linear(hidden_size* 2, 1, bias=False)

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden1(x2))
        x = torch.cat((h1, h2), dim=1)
        z = self.sigmoid_f(self.hidden_sigmoid(x))

        return z.view(z.size()[0], 1) * h1 + (1 - z).view(z.size()[0], 1) * h2

class GatedOutputLayer(nn.Module):
    """
    Gated Multimodal Layer based on 'Gated multimodal networks,
    Arevalo1 et al.' (https://arxiv.org/abs/1702.01992)
    """

    def __init__(self, hidden_size=3):
        super(GatedOutputLayer, self).__init__()

        self.hidden1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_sigmoid = nn.Linear(hidden_size* 2, 1, bias=False)

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden1(x2))
        x = torch.cat((h1, h2), dim=1)
        z = self.sigmoid_f(self.hidden_sigmoid(x))


        return z.view(z.size()[0], 1) * h1 + (1 - z).view(z.size()[0], 1) * h2  , z.mean().detach()

if __name__ == '__main__':
        # Example: Image and text features are randomly initialized for demonstration
    hidden_size = 256
    model = EnhancedFilterModule(hidden_size)

    # Random image and text features for demonstration
    img_features = torch.randn(8, hidden_size)
    txt_features = torch.randn(8, hidden_size)

    out_put=model(img_features)

    print(out_put.shape)

    model_fus= GatedMultimodalLayer(hidden_size,hidden_size,hidden_size)

    out_put_fus=model_fus(img_features,txt_features)

    print(out_put_fus.shape)

    # # Get fused features
    # fused_features = model(img_features, txt_features)
    # print(fused_features.shape)
