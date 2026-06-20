#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Attribute-guided breast ultrasound image classification model
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


# REMOVED: GeometricAttention class - no longer used (simplified model without geometric attention)
# REMOVED: Similarity threshold - no longer used (simplified model without threshold)


class FetalAttributeClassifier(nn.Module):
    """
    属性引导的乳腺超声图像分类模型
    
    核心机制:
    1. 视觉特征提取 (ResNet/ViT backbone)
    2. 属性预测 (通过patch级别的余弦相似度)
    3. 属性到类别的转换 (使用类别-属性映射矩阵)
    4. 直接分类预测 (全连接层)
    5. 融合两个分类预测
    """
    
    def __init__(
        self,
        backbone_type='resnet50',
        num_classes=3,
        num_attrs=50,
        attr_emb_dim=512,
        backbone_feat_dim=2048,
        common_dim=512,
        fusion_weight=0.7,
        resnet_path=None,
        temperature=1.0,
        grid_size=14,
        dropout=0.1,
    ):
        """
        Args:
            backbone_type: Backbone网络类型 ('resnet50', 'resnet101', 或 'vitbase')
            num_classes: 类别数量 (3 for breast: benign, malignant, normal)
            num_attrs: 属性数量
            attr_emb_dim: 属性嵌入维度
            backbone_feat_dim: Backbone特征维度 (ResNet50:2048, ResNet101:2048, ViT-Base:768)
            common_dim: 视觉和属性特征的公共投影维度
            fusion_weight: 直接分类的融合权重 (0-1)
            resnet_path: 预训练backbone权重路径
            temperature: 相似度缩放温度参数
            grid_size: 网格大小（用于ViT）
            dropout: 分类头Dropout率（与baseline一致，可由--dropout传入）
        """
        super(FetalAttributeClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.num_attrs = num_attrs
        self.attr_emb_dim = attr_emb_dim
        self.common_dim = common_dim
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.grid_size = grid_size
        
        # 1. Backbone网络
        self.backbone = self._build_backbone(backbone_type, resnet_path)
        
        # 2. 投影层：对齐视觉和属性特征维度
        self.proj_vis = nn.Sequential(
            nn.Linear(backbone_feat_dim, common_dim),
            nn.Dropout(0.4)
        )
        self.proj_attr = nn.Sequential(
            nn.Linear(attr_emb_dim, common_dim),
            nn.Dropout(0.2)
        )
        
        # 3. 直接分类分支（dropout 与 baseline 一致，由 args.dropout 传入）
        self.cls_fc = nn.Linear(backbone_feat_dim, num_classes)
        self.cls_dropout = nn.Dropout(dropout)
        
        # 4. SIMPLIFIED: 移除可学习阈值和几何注意力机制
        # 不再使用相似度阈值和几何注意力，直接使用原始相似度
        
        # 6. 类别-属性映射矩阵
        self.register_buffer(
            "class_attr_map",
            nn.init.normal_(torch.empty(num_classes, num_attrs), std=0.01)
        )
        
        # 7. 融合权重（可学习，使用sigmoid约束到[0,1]）
        self.fusion_weight = nn.Parameter(torch.tensor(fusion_weight))
    
    def _build_backbone(self, backbone_type, resnet_path):
        """构建backbone网络"""
        print(f"[Model] Building {backbone_type.upper()} backbone...")
        
        if backbone_type == 'resnet50':
            backbone = models.resnet50(weights=None)
            if resnet_path is not None and os.path.exists(resnet_path):
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
                
                backbone_state = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
                missing_keys, unexpected_keys = backbone.load_state_dict(backbone_state, strict=False)
                if missing_keys:
                    print(f"[Model] Warning: Missing keys when loading weights: {missing_keys[:5]}...")
                if unexpected_keys:
                    print(f"[Model] Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
                print(f"[Model] ResNet50 weights loaded successfully!")
            else:
                if resnet_path is not None:
                    print(f"[Model] Warning: ResNet50 weights file not found at {resnet_path}, using random initialization")
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            print(f"[Model] ResNet50 backbone ready (feature dimension: 2048)")
            
        elif backbone_type == 'resnet101':
            backbone = models.resnet101(weights=None)
            if resnet_path is not None and os.path.exists(resnet_path):
                print(f"[Model] Loading ResNet101 weights: {resnet_path}")
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
                
                backbone_state = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
                missing_keys, unexpected_keys = backbone.load_state_dict(backbone_state, strict=False)
                if missing_keys:
                    print(f"[Model] Warning: Missing keys when loading weights: {missing_keys[:5]}...")
                if unexpected_keys:
                    print(f"[Model] Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
                print(f"[Model] ResNet101 weights loaded successfully!")
            else:
                if resnet_path is not None:
                    print(f"[Model] Warning: ResNet101 weights file not found at {resnet_path}, using random initialization")
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            print(f"[Model] ResNet101 backbone ready (feature dimension: 2048)")
            
        elif backbone_type == 'vitbase':
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
            raise ValueError(f"Unsupported backbone type: {backbone_type}. Only 'resnet50', 'resnet101', and 'vitbase' are supported.")
        
        return backbone
    
    def ClsBySimilar(self, x_vis_patches, attr_protos):
        """
        通过patch级别的余弦相似度计算属性预测（简化版本，无阈值，无几何注意力）
        
        Args:
            x_vis_patches: (batch_size, num_patches, common_dim) - 投影后的视觉patch特征
            attr_protos: (num_attrs, common_dim) - 投影后的属性原型
        
        Returns:
            attr_logits: (batch_size, num_attrs) - 属性预测分数
            patch_similarity: (batch_size, num_attrs, num_patches) - 原始patch相似度
        """
        batch_size, num_patches, _ = x_vis_patches.shape
        
        # 归一化
        x_vis_patches_norm = F.normalize(x_vis_patches, p=2, dim=2)  # (B, N, common_dim)
        attr_protos_norm = F.normalize(attr_protos, p=2, dim=1)  # (num_attrs, common_dim)
        
        # 计算每个patch与每个属性的余弦相似度（原始相似度，无阈值，无几何注意力）
        attr_protos_expanded = attr_protos_norm.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2)  # (B, common_dim, num_attrs)
        cosine_sim = torch.bmm(x_vis_patches_norm, attr_protos_expanded)  # (B, N, num_attrs)
        patch_similarity = cosine_sim.permute(0, 2, 1)  # (B, num_attrs, N) - 原始相似度
        
        # SIMPLIFIED: 使用简单的加权平均聚合（无阈值，无几何注意力）
        # 使用softmax加权，让高相似度区域权重更大
        similarity_weights = F.softmax(patch_similarity * 2.0, dim=2)  # (B, num_attrs, N)
        attr_logits = (patch_similarity * similarity_weights).sum(dim=2)  # (B, num_attrs) - 加权平均
        
        # 使用温度缩放
        attr_logits = attr_logits / (self.temperature * 1.2)
        
        return attr_logits, patch_similarity
    
    def attribute_to_class(self, attr_embed, patch_similarity=None):
        """
        将属性嵌入转换为分类预测（简化版本，只使用属性分数作为权重）
        
        Args:
            attr_embed: (batch_size, num_attrs) - 属性预测分数/概率
            patch_similarity: (batch_size, num_attrs, num_patches) - patch级别的相似度（可选，当前不使用）
        
        Returns:
            attr_cls_logits: (batch_size, num_classes) - 从属性转换的分类预测
        """
        batch_size = attr_embed.shape[0]
        num_classes = self.class_attr_map.shape[0]
        device = attr_embed.device
        
        # SIMPLIFIED: 只使用属性分数作为权重（无阈值，无几何注意力）
        attr_weights = attr_embed  # (B, num_attrs) - 简单的属性分数
        
        attr_cls_logits = torch.zeros(batch_size, num_classes, device=device)
        
        for class_idx in range(num_classes):
            class_attrs_mask = self.class_attr_map[class_idx]  # (num_attrs,)
            num_class_attrs = class_attrs_mask.sum().item()
            
            if num_class_attrs > 0:
                masked_attr_scores = attr_embed * class_attrs_mask.unsqueeze(0)  # (batch_size, num_attrs)
                masked_weights = attr_weights * class_attrs_mask.unsqueeze(0)  # (batch_size, num_attrs)
                
                # 使用加权平均：高分属性权重更大
                weighted_sum = (masked_attr_scores * masked_weights).sum(dim=1)  # (batch_size,)
                weight_sum = masked_weights.sum(dim=1)  # (batch_size,)
                class_score = weighted_sum / (weight_sum + 1e-8)  # (batch_size,)
                
                attr_cls_logits[:, class_idx] = class_score
            else:
                attr_cls_logits[:, class_idx] = -1e6
        
        # 缩放到logits空间，匹配分支一的logits尺度
        attr_cls_logits = (attr_cls_logits - 0.5) * 20.0  # 范围从[-5, 5]扩展到[-10, 10]
        
        return attr_cls_logits
    
    def forward(self, x, attrs_matrix):
        """
        前向传播
        
        Args:
            x: (batch_size, 3, H, W) - 输入图像
            attrs_matrix: (num_attrs, attr_emb_dim) - 所有属性的嵌入矩阵
        
        Returns:
            dict: 包含各种预测结果的字典
        """
        # 1. 视觉特征提取
        x_vis_raw = self.backbone(x)
        
        # 处理不同backbone的输出格式
        if len(x_vis_raw.shape) == 4:
            # ResNet 4D输出: (batch_size, channels, H, W)
            x_vis = F.adaptive_avg_pool2d(x_vis_raw, (1, 1)).squeeze(-1).squeeze(-1)  # (B, C)
            # 获取patch特征（用于分支二）
            B, C, H, W = x_vis_raw.shape
            x_vis_patches_raw = x_vis_raw.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        elif len(x_vis_raw.shape) == 3:
            # ViT 3D输出: (batch_size, num_patches+1, backbone_feat_dim)
            x_vis = x_vis_raw[:, 0, :]  # (B, C) - [CLS] token
            x_vis_patches_raw = x_vis_raw[:, 1:, :]  # (B, N, C) - patch tokens
        else:
            # ResNet 2D输出: (batch_size, backbone_feat_dim)
            x_vis = x_vis_raw
            x_vis_patches_raw = x_vis.unsqueeze(1)  # (B, 1, C)
        
        # 2. 投影属性嵌入到公共维度
        assert attrs_matrix.shape[0] == self.num_attrs, f"Expected {self.num_attrs} attributes, got {attrs_matrix.shape[0]}"
        assert attrs_matrix.shape[1] == self.attr_emb_dim, f"Expected embedding dim {self.attr_emb_dim}, got {attrs_matrix.shape[1]}"
        
        x_attr = self.proj_attr(attrs_matrix)  # (num_attrs, common_dim)
        x_vis_proj = self.proj_vis(x_vis)      # (batch_size, common_dim)
        
        # 归一化
        x_attr = F.normalize(x_attr, p=2, dim=-1)
        x_vis_proj = F.normalize(x_vis_proj, p=2, dim=-1)
        
        # 投影patch特征到公共维度
        B, N, C = x_vis_patches_raw.shape
        x_vis_patches_flat = x_vis_patches_raw.contiguous().view(B * N, C)
        x_vis_patches_proj_flat = self.proj_vis(x_vis_patches_flat)  # (B*N, common_dim)
        x_vis_patches_proj = x_vis_patches_proj_flat.view(B, N, self.common_dim)
        x_vis_patches_proj = F.normalize(x_vis_patches_proj, p=2, dim=2)
        
        # ========== 分支一: 直接视觉分类预测 ==========
        cls_logits = self.cls_fc(x_vis)  # (batch_size, num_classes)
        cls_logits = self.cls_dropout(cls_logits)
        
        # ========== 分支二: 属性预测（基于patch相似度） ==========
        attr_logits, patch_similarity = self.ClsBySimilar(x_vis_patches_proj, x_attr)  # (B, num_attrs), (B, num_attrs, N)
        attr_probs = torch.sigmoid(attr_logits)  # (batch_size, num_attrs)
        
        # 将属性预测转换为分类预测（简化版本，只使用属性分数）
        attr_cls_logits = self.attribute_to_class(attr_probs, patch_similarity=patch_similarity)  # (batch_size, num_classes)
        
        # ========== 融合: 结合两个分支 ==========
        fusion_weight = torch.sigmoid(self.fusion_weight)
        
        # 修复：直接在logits空间融合，避免尺度不匹配问题
        # 先对logits进行温度缩放，确保两个分支的尺度接近
        cls_logits_scaled = cls_logits / self.temperature
        attr_cls_logits_scaled = attr_cls_logits / self.temperature
        
        # 加权融合logits
        alpha = fusion_weight
        beta = 1 - fusion_weight
        fus_logits = alpha * cls_logits_scaled + beta * attr_cls_logits_scaled
        
        # 返回所有预测
        result = {
            'attr_logits': attr_logits,
            'cls_logits': cls_logits,
            'attr_cls_logits': attr_cls_logits,
            'fus_logits': fus_logits,
            'vis_embed': x_vis_proj,
            'attr_embed': attr_probs,
        }
        
        # CRITICAL: 输出patch_similarity，确保可视化使用与训练完全相同的相似度
        # patch_similarity是原始余弦相似度（无阈值，无几何注意力），与训练时计算的一致
        result['patch_similarity'] = patch_similarity  # (B, num_attrs, N) - 原始相似度
        result['x_vis_patches_proj'] = x_vis_patches_proj  # (B, N, common_dim)
        result['x_attr'] = x_attr  # (num_attrs, common_dim)
        
        # 添加原始视觉特征（用于可视化，可选）
        if len(x_vis_raw.shape) == 3:
            result['x_vis_raw'] = x_vis_raw  # ViT: (B, N+1, C)
        
        # SIMPLIFIED: 不再输出几何注意力和阈值信息
        
        return result
