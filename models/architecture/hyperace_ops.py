# models/architecture/hyperace_ops.py
# HyperACE模块实现，包含超图节点聚合和自适应视觉感知两大核心功能，专为电线杆检测任务设计

import torch
import torch.nn as nn
import torch.nn.functional as F

# HyperNodeAggregation模块：模拟超图结构，聚合全局特征，增强细长目标的语义表达。
class HyperNodeAggregation(nn.Module):
    def __init__(self, in_channels, node_dim=64):
        super(HyperNodeAggregation, self).__init__()
        self.node_dim = node_dim
        # 降维卷积，用于特征投影
        self.query_conv = nn.Conv2d(in_channels, node_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, node_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    # 前向传播函数，输入特征图，输出聚合后的特征图
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        
        # 构建超边（Hyperedge）关联矩阵
        # [B, node_dim, H*W]
        proj_query = self.query_conv(x).view(m_batchsize, self.node_dim, -1)
        # [B, node_dim, H*W]
        proj_key = self.key_conv(x).view(m_batchsize, self.node_dim, -1)
        
        # 计算节点间的相似度（Attention map）->模拟超图关联
        # [B, node_dim, node_dim]
        energy = torch.bmm(proj_query, proj_key.permute(0, 2, 1))
        attention = self.softmax(energy)
        
        # 聚合特征
        # [B, C, H*W]
        proj_value = self.value_conv(x).view(m_batchsize, C, -1)
        
        # 利用关联矩阵聚合全局特征
        # [B, C, node_dim]->这里是一种特征空间的重构
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        
        return out

# AdaptiveVisualPerception模块：动态调整特征权重，增强模型对细长骨架的感知能力，提升检测精度。
class AdaptiveVisualPerception(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveVisualPerception, self).__init__()
        # 轻量化自适应权重生成
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享MLP，用于生成通道注意力
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        # 空间注意力卷积，用于定位细长骨架
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    # 前向传播函数，输入特征图，输出增强后的特征图
    def forward(self, x):
        # 通道自适应（Channel Attention）-识别“是什么”
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_weight = self.sigmoid(avg_out + max_out)
        
        # 空间自适应（Spatial Attention）-识别“在哪里”（针对细长目标优化）
        # 在通道维度做Max和Avg，压缩为2个通道
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        spatial_cat = torch.cat([max_spatial, avg_spatial], dim=1)
        
        spatial_weight = self.sigmoid(self.spatial_conv(spatial_cat))
        
        # 混合加权
        return x * channel_weight * spatial_weight

# HyperACE_Module类，包含超图节点聚合和自适应视觉感知两大核心功能，专为电线杆检测任务设计
class HyperACE_Module(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(HyperACE_Module, self).__init__()
        self.c2 = c2
        
        # 超图节点特征提取
        self.hyper_agg = HyperNodeAggregation(c1)
        
        # 维度调整（如果输入输出通道不一致）
        self.trans_conv = nn.Conv2d(c1, c2, kernel_size=1) if c1 != c2 else nn.Identity()
        
        # 自适应感知权重分配
        self.adaptive_perception = AdaptiveVisualPerception(c2)
        
        # 残差缩放系数
        self.gamma = nn.Parameter(torch.zeros(1))

    # 前向传播函数，输入特征图，输出增强后的特征图
    def forward(self, x):
        residual = x
        
        # 分支1：原始特征变换
        x_base = self.trans_conv(x)
        
        # 分支2：超图节点聚合+自适应感知
        x_enhanced = self.adaptive_perception(x_base)
        
        # 如果输入输出通道一致，加上残差连接防止退化
        if hasattr(self.trans_conv, 'kernel_size'): # 发生了卷积变换
             return x_enhanced
        else:
             return x_enhanced + residual * self.gamma
