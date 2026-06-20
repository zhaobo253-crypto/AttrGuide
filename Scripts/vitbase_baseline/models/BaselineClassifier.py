#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline breast ultrasound image classification model (ViT only, no attribute guidance)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BaselineClassifier(nn.Module):
    """
    基线乳腺超声图像分类模型（只使用ViT，无属性引导）
    
    核心机制:
    1. 视觉特征提取 (ViT backbone)
    2. 直接分类预测 (全连接层)
    """
    
    def __init__(
        self,
        backbone_type='vitbase',
        num_classes=3,
        backbone_feat_dim=768,
        resnet_path=None,
        dropout=0.1,
    ):
        """
        Args:
            backbone_type: Backbone网络类型 ('vitbase' only for baseline)
            num_classes: 类别数量 (3 for breast: benign, malignant, normal)
            backbone_feat_dim: Backbone特征维度 (ViT-Base:768)
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
        
        if backbone_type == 'vitbase':
            vit_model = models.vit_b_16(weights=None)
            if resnet_path is not None and os.path.exists(resnet_path):
                print(f"[Model] Loading ViT-Base weights: {resnet_path}")
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
                
                backbone_state = {k: v for k, v in state_dict.items() if not k.startswith('heads.')}
                missing_keys, unexpected_keys = vit_model.load_state_dict(backbone_state, strict=False)
                if missing_keys:
                    print(f"[Model] Warning: Missing keys when loading weights: {missing_keys[:5]}...")
                if unexpected_keys:
                    print(f"[Model] Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
                print(f"[Model] ViT-Base weights loaded successfully!")
            else:
                if resnet_path is not None:
                    print(f"[Model] Warning: ViT-Base weights file not found at {resnet_path}, using random initialization")
            
            vit_model.heads = None
            
            def vit_forward_without_heads(self, x):
                n = self.conv_proj.kernel_size[0]
                x = self.conv_proj(x)
                x = x.flatten(2).transpose(1, 2)
                batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                
                if hasattr(self, 'encoder_pos_embedding'):
                    x = x + self.encoder_pos_embedding
                elif hasattr(self, 'pos_embedding'):
                    x = x + self.pos_embedding
                
                x = self.encoder(x)
                return x
            
            import types
            vit_model.forward = types.MethodType(vit_forward_without_heads, vit_model)
            backbone = vit_model
            print(f"[Model] ViT-Base backbone ready (feature dimension: 768)")
        else:
            raise ValueError(f"Baseline model only supports 'vitbase' backbone, got: {backbone_type}")
        
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
        x_vis_raw = self.backbone(x)
        
        # 处理ViT输出: (batch_size, num_patches+1, backbone_feat_dim)
        if len(x_vis_raw.shape) == 3:
            # ViT 3D输出: [CLS] token是第一个
            x_vis = x_vis_raw[:, 0, :]  # (B, C) - [CLS] token
        else:
            # 其他格式（不应该出现）
            raise ValueError(f"Unexpected backbone output shape: {x_vis_raw.shape}")
        
        # 2. 直接分类预测
        cls_logits = self.cls_fc(x_vis)  # (batch_size, num_classes)
        cls_logits = self.cls_dropout(cls_logits)
        
        # 返回结果
        result = {
            'cls_logits': cls_logits,  # 唯一输出（用于baseline）
            'vis_embed': x_vis,  # 视觉特征（可选，用于可视化）
        }
        
        return result
