#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline breast ultrasound image classification model (ResNet50 only, no attribute guidance)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BaselineClassifier(nn.Module):
    """
    基线乳腺超声图像分类模型（只使用ResNet50，无属性引导）

    核心机制:
    1. 视觉特征提取 (ResNet50 backbone)
    2. 直接分类预测 (全连接层)
    """
    
    def __init__(
        self,
        backbone_type='resnet50',
        num_classes=3,
        backbone_feat_dim=2048,
        resnet_path=None,
        dropout=0.1,
    ):
        """
        Args:
            backbone_type: Backbone网络类型 ('resnet50' only for baseline)
            num_classes: 类别数量 (3 for breast: benign, malignant, normal)
            backbone_feat_dim: Backbone特征维度 (ResNet50:2048)
            resnet_path: 预训练backbone权重路径
            dropout: Dropout率
        """
        super(BaselineClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # 1. Backbone网络
        self.backbone = self._build_backbone(backbone_type, resnet_path)
        
        # 2. 分类头
        self.cls_fc = nn.Linear(backbone_feat_dim, num_classes)
        self.cls_dropout = nn.Dropout(dropout)
    
    def _build_backbone(self, backbone_type, resnet_path):
        """构建backbone网络"""
        print(f"[Model] Building {backbone_type.upper()} backbone...")

        if backbone_type == 'resnet50':
            # 创建ResNet50模型
            resnet_model = models.resnet50(weights=None)

            # 直接加载预训练权重
            print(f"[Model] Loading ResNet50 weights: {resnet_path}")
            checkpoint = torch.load(resnet_path, map_location='cpu')

            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            if not hasattr(state_dict, 'items'):
                raise TypeError(f"Checkpoint format not supported. Expected dict or state_dict, got {type(checkpoint)}")

            # 移除分类器相关的权重
            backbone_state = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
            missing_keys, unexpected_keys = resnet_model.load_state_dict(backbone_state, strict=False)
            if missing_keys:
                print(f"[Model] Warning: Missing keys when loading weights: {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"[Model] Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
            print(f"[Model] ResNet50 weights loaded successfully!")

            # 移除最后的分类层
            resnet_model.fc = nn.Identity()
            backbone = resnet_model
            print(f"[Model] ResNet50 backbone ready (feature dimension: 2048)")
        else:
            raise ValueError(f"Baseline model only supports 'resnet50' backbone, got: {backbone_type}")

        return backbone
    
    def forward(self, x):
        """
        前向传播

        Args:
            x: (batch_size, 3, H, W) - 输入图像

        Returns:
            dict: 包含分类预测结果的字典
        """
        # 1. 视觉特征提取
        x_vis = self.backbone(x)  # ResNet50输出: (batch_size, 2048)

        # 处理ResNet50输出: (batch_size, backbone_feat_dim)
        if len(x_vis.shape) != 2:
            raise ValueError(f"Unexpected ResNet50 backbone output shape: {x_vis.shape}")

        # 2. 直接分类预测
        cls_logits = self.cls_fc(x_vis)  # (batch_size, num_classes)
        cls_logits = self.cls_dropout(cls_logits)

        # 返回结果
        result = {
            'cls_logits': cls_logits,  # 唯一输出（用于baseline）
            'vis_embed': x_vis,  # 视觉特征（可选，用于可视化）
        }

        return result
