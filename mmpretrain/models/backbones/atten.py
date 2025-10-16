import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


# 定义注意力模块，threshold 和 K 均为可学习参数
class LearnableGlobalAttention(nn.Module):
    def __init__(self, channels, reduction=16, init_threshold=1.2, init_k=10.0):
        """
        :param channels: 输入特征通道数
        :param reduction: 用于全连接层的降维率
        :param init_threshold: 阈值初始值（当局部特征值超过均值的比例）
        :param init_k: Sigmoid 中控制陡峭程度的初始值
        threshold阈值：局部值与均值之比。init_threshold=1.2,意味着当局部特征值超过其均值 1.2 倍时，就可能认定为异常区域
        """
        super(LearnableGlobalAttention, self).__init__()
        self.channels = channels
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        # 将 threshold 和 K 设为可学习参数
        self.threshold = nn.Parameter(torch.tensor(init_threshold, dtype=torch.float))
        self.k = nn.Parameter(torch.tensor(init_k, dtype=torch.float))

    def forward(self, x):
        """
        :param x: 输入特征图 [B, C, H, W]
        :return: 加权后的特征图及注意力掩码（用于可视化）
        """
        B, C, H, W = x.size()

        # 全局平均池化得到全局均值向量
        global_mean = F.adaptive_avg_pool2d(x, (1, 1)).view(B, C)
        attn_global = F.relu(self.fc1(global_mean))
        attn_global = torch.sigmoid(self.fc2(attn_global)).view(B, C, 1, 1)

        # 计算局部均值（这里简单使用空间平均）
        eps = 1e-6
        local_mean = x.mean(dim=(2, 3), keepdim=True) + eps
        ratio = x / local_mean  # 得到局部与均值的比值

        # 使用软阈值：输出范围 [0, 1]，接近 1 表示正常区域，接近 0 表示缺陷区域
        defect_mask = torch.sigmoid(self.k * (self.threshold - ratio))

        # 融合全局信息与缺陷抑制
        attn = attn_global * defect_mask

        # 返回加权后的特征图和用于可视化的掩码
        out = x * attn
        return out, defect_mask


# 简单的特征提取器（例如一层卷积），模拟 ResNet 的前几层
class SimpleFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(SimpleFeatureExtractor, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# 整合特征提取与注意力模块
class AttentionVisualizationNet(nn.Module):
    def __init__(self):
        super(AttentionVisualizationNet, self).__init__()
        self.feature_extractor = SimpleFeatureExtractor(in_channels=3, out_channels=64)
        self.attention = LearnableGlobalAttention(channels=64, reduction=16, init_threshold=1.2, init_k=10.0)

    def forward(self, x):
        features = self.feature_extractor(x)  # 提取特征图
        attn_features, mask = self.attention(features)  # 应用注意力
        return attn_features, mask


# 加载图片并预处理
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((282, 282)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # 增加 batch 维度


# 可视化注意力掩码
def visualize_attention(mask, save_path='attention_heatmap.png'):
    # 假设 mask 的形状为 [B, C, H, W]，这里选取第一个样本、取平均通道得到 [H, W]
    attn_map = mask[0].mean(dim=0).detach().cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(attn_map, cmap='jet')
    plt.colorbar()
    plt.title('Attention Mask')
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    # 图片路径（请确保路径有效）
    img_path = r'C:/Users/lenovo/Desktop/模型消融/LearnableGlobalAttention可视化/1.bmp'
    image_tensor = load_image(img_path)

    # 构建网络并前向传播
    net = AttentionVisualizationNet()
    attn_features, mask = net(image_tensor)

    # 输出当前的可学习参数值
    print("Learned threshold:", net.attention.threshold.item())
    print("Learned k:", net.attention.k.item())

    # 可视化注意力掩码
    visualize_attention(mask, save_path='attention_heatmap.png')
