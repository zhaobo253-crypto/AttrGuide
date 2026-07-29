#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Loss function definitions
Reference: TransZero loss function design
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttributeGuidedLoss(nn.Module):
    """
    Attribute-guided multi-task loss function
    Reference: TransZero loss function design
    """
    
    def __init__(
        self,
        lambda_fus: float = 0.5,
        lambda_reg: float = 0.3,
        lambda_attr_pred: float = 0.2,
        train_mode: str = "fus",
    ):
        """
        Args:
            lambda_fus: Weight for fusion result loss
            lambda_reg: Weight for regularization loss
            lambda_attr_pred: Weight for attribute prediction loss (multi-label binary classification)
            train_mode: Training mode - "attr_only", "cls_only", "fus", or "avg_fus"
        """
        super(AttributeGuidedLoss, self).__init__()
        self.lambda_fus = lambda_fus
        self.lambda_reg = lambda_reg
        self.lambda_attr_pred = lambda_attr_pred
        self.train_mode = train_mode
    
    def forward(
        self,
        outputs: dict,
        labels: torch.Tensor,
        attr_labels: Optional[torch.Tensor] = None,
        class_attr_map: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            outputs: Model output dictionary
                - attr_logits: (batch_size, num_attrs) - Attribute prediction
                - cls_logits: (batch_size, num_classes) - Direct classification prediction
                - attr_cls_logits: (batch_size, num_classes) - Attribute-guided classification prediction
                - fus_logits: (batch_size, num_classes) - Fused classification prediction
                - vis_embed: (batch_size, common_dim) - Visual embedding (for regularization, optional)
                - attr_embed: (batch_size, num_attrs) - Normalized attribute predictions (for regularization, optional)
            labels: (batch_size,) - Class labels
            attr_labels: (batch_size, num_attrs) - Attribute labels (optional)
            class_attr_map: (num_classes, num_attrs) - Class-attribute mapping matrix (for regularization)
        
        Returns:
            loss_dict: Dictionary containing all loss components
        """
        attr_cls_logits = outputs.get('attr_cls_logits', None)
        cls_logits = outputs.get('cls_logits', None)
        fus_logits = outputs.get('fus_logits', None)
        attr_logits = outputs['attr_logits']
        
        # Initialize all losses to zero
        loss_fus = torch.tensor(0.0, device=labels.device)
        loss_attr_cls = torch.tensor(0.0, device=labels.device)
        loss_cls = torch.tensor(0.0, device=labels.device)
        loss_attr_pred = torch.tensor(0.0, device=labels.device)
        loss_reg = torch.tensor(0.0, device=labels.device)
        
        # Calculate losses based on training mode
        if self.train_mode == "attr_only":
            # Only train attribute branch - use attr_cls_logits for classification loss
            if attr_cls_logits is not None:
                loss_attr_cls = F.cross_entropy(attr_cls_logits, labels)
            # Attribute prediction loss
            if attr_labels is not None:
                loss_attr_pred = F.binary_cross_entropy_with_logits(attr_logits, attr_labels.float())
            # Regularization loss
            if class_attr_map is not None:
                target_attrs = class_attr_map[labels]
                if 'attr_embed' in outputs:
                    attr_embed = outputs['attr_embed']
                else:
                    attr_embed = torch.sigmoid(attr_logits)
                attr_embed_norm = F.normalize(attr_embed, p=2, dim=-1)
                target_attrs_norm = F.normalize(target_attrs, p=2, dim=-1)
                loss_reg_mse = F.mse_loss(attr_embed, target_attrs, reduction='mean')
                loss_reg_cos = 1.0 - (attr_embed_norm * target_attrs_norm).sum(dim=-1).mean()
                loss_reg = loss_reg_mse + 0.2 * loss_reg_cos
            # Total loss for attr_only mode
            loss = self.lambda_attr_pred * loss_attr_pred + self.lambda_reg * loss_reg
            if attr_cls_logits is not None:
                loss = loss + loss_attr_cls  # Use attr_cls loss as main classification loss
                
        elif self.train_mode == "cls_only":
            # Only train classification head - use cls_logits for classification loss
            if cls_logits is not None:
                loss_cls = F.cross_entropy(cls_logits, labels)
            # Total loss for cls_only mode
            loss = loss_cls
            
        elif self.train_mode == "fus":
            # Train with weighted fusion
            if fus_logits is not None:
                loss_fus = F.cross_entropy(fus_logits, labels)
            # Attribute prediction loss
            if attr_labels is not None:
                loss_attr_pred = F.binary_cross_entropy_with_logits(attr_logits, attr_labels.float())
            # Regularization loss
            if class_attr_map is not None:
                target_attrs = class_attr_map[labels]
                if 'attr_embed' in outputs:
                    attr_embed = outputs['attr_embed']
                else:
                    attr_embed = torch.sigmoid(attr_logits)
                attr_embed_norm = F.normalize(attr_embed, p=2, dim=-1)
                target_attrs_norm = F.normalize(target_attrs, p=2, dim=-1)
                loss_reg_mse = F.mse_loss(attr_embed, target_attrs, reduction='mean')
                loss_reg_cos = 1.0 - (attr_embed_norm * target_attrs_norm).sum(dim=-1).mean()
                loss_reg = loss_reg_mse + 0.2 * loss_reg_cos
            # Total loss for fus mode
            loss = self.lambda_fus * loss_fus + self.lambda_reg * loss_reg
            if attr_labels is not None:
                loss = loss + self.lambda_attr_pred * loss_attr_pred
                
        elif self.train_mode == "avg_fus":
            # Train both branches but use average fusion for loss
            if attr_cls_logits is not None and cls_logits is not None:
                avg_fus_logits = (cls_logits + attr_cls_logits) / 2.0
                loss_fus = F.cross_entropy(avg_fus_logits, labels)
            # Attribute prediction loss
            if attr_labels is not None:
                loss_attr_pred = F.binary_cross_entropy_with_logits(attr_logits, attr_labels.float())
            # Regularization loss
            if class_attr_map is not None:
                target_attrs = class_attr_map[labels]
                if 'attr_embed' in outputs:
                    attr_embed = outputs['attr_embed']
                else:
                    attr_embed = torch.sigmoid(attr_logits)
                attr_embed_norm = F.normalize(attr_embed, p=2, dim=-1)
                target_attrs_norm = F.normalize(target_attrs, p=2, dim=-1)
                loss_reg_mse = F.mse_loss(attr_embed, target_attrs, reduction='mean')
                loss_reg_cos = 1.0 - (attr_embed_norm * target_attrs_norm).sum(dim=-1).mean()
                loss_reg = loss_reg_mse + 0.2 * loss_reg_cos
            # Total loss for avg_fus mode
            loss = self.lambda_fus * loss_fus + self.lambda_reg * loss_reg
            if attr_labels is not None:
                loss = loss + self.lambda_attr_pred * loss_attr_pred
        
        return {
            'loss': loss,
            'loss_fus': loss_fus,
            'loss_attr_cls': loss_attr_cls,
            'loss_cls': loss_cls,
            'loss_attr_pred': loss_attr_pred,
            'loss_reg': loss_reg,
        }


def compute_classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy"""
    preds = logits.argmax(dim=1)
    correct = (preds == labels).float().sum()
    return (correct / labels.size(0)).item()


def compute_per_class_accuracy(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    """Compute per-class accuracy from logits"""
    preds = logits.argmax(dim=1)
    per_class_acc = {}
    
    for class_idx in range(num_classes):
        mask = labels == class_idx
        if mask.any():
            class_correct = (preds[mask] == labels[mask]).float().sum()
            class_total = mask.float().sum()
            per_class_acc[class_idx] = (class_correct / class_total).item()
    
    return per_class_acc


def compute_per_class_accuracy_from_predictions(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    """Compute per-class accuracy from predictions (already argmaxed)"""
    # Ensure preds and labels are 1D tensors and have correct dtype
    preds = preds.flatten().long()  # Flatten to 1D and convert to long
    labels = labels.flatten().long()  # Flatten to 1D and convert to long
    
    # Ensure they have the same length
    assert len(preds) == len(labels), f"Predictions and labels must have the same length, got {len(preds)} and {len(labels)}"
    
    per_class_acc = {}
    
    for class_idx in range(num_classes):
        mask = labels == class_idx
        if mask.any():
            class_correct = (preds[mask] == labels[mask]).float().sum()
            class_total = mask.float().sum()
            per_class_acc[class_idx] = (class_correct / class_total).item()
    
    return per_class_acc


def compute_attribute_prediction_accuracy(attr_logits: torch.Tensor, attr_labels: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute attribute prediction accuracy (multi-label binary classification accuracy)
    
    Args:
        attr_logits: (batch_size, num_attrs) - Attribute prediction logits
        attr_labels: (batch_size, num_attrs) - Ground truth attribute labels (0 or 1)
        threshold: Threshold for converting logits to binary predictions (default: 0.5)
    
    Returns:
        accuracy: Overall attribute prediction accuracy (percentage of correct attribute predictions)
    """
    # Convert logits to probabilities
    attr_probs = torch.sigmoid(attr_logits)  # (batch_size, num_attrs)
    
    # Convert to binary predictions
    attr_preds = (attr_probs >= threshold).float()  # (batch_size, num_attrs)
    
    # Ensure labels are float
    attr_labels_float = attr_labels.float()
    
    # Compute accuracy: percentage of correct attribute predictions across all samples and attributes
    correct = (attr_preds == attr_labels_float).float()
    accuracy = correct.mean().item()
    
    return accuracy
