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
        lambda_attr: float = 0.3,
        lambda_cls: float = 0.3,
        lambda_fus: float = 0.1,
        lambda_reg: float = 0.1,
        lambda_attr_pred: float = 0.1,
    ):
        """
        Args:
            lambda_attr: Weight for attribute-guided classification loss
            lambda_cls: Weight for direct classification loss
            lambda_fus: Weight for fusion result loss
            lambda_reg: Weight for regularization loss
            lambda_attr_pred: Weight for attribute prediction loss (multi-label binary classification)
        """
        super(AttributeGuidedLoss, self).__init__()
        self.lambda_attr = lambda_attr
        self.lambda_cls = lambda_cls
        self.lambda_fus = lambda_fus
        self.lambda_reg = lambda_reg
        self.lambda_attr_pred = lambda_attr_pred
    
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
        attr_cls_logits = outputs['attr_cls_logits']
        cls_logits = outputs['cls_logits']
        fus_logits = outputs['fus_logits']
        attr_logits = outputs['attr_logits']
        
        # 1. Attribute-guided classification loss (Branch 2)
        loss_attr = F.cross_entropy(attr_cls_logits, labels)
        
        # 2. Direct classification loss (Branch 1)
        loss_cls = F.cross_entropy(cls_logits, labels)
        
        # 3. Fusion result loss
        loss_fus = F.cross_entropy(fus_logits, labels)
        
        # 4. Attribute prediction loss (if attribute labels are available)
        loss_attr_pred = torch.tensor(0.0, device=labels.device)
        if attr_labels is not None:
            # Use binary cross entropy (each attribute is binary)
            loss_attr_pred = F.binary_cross_entropy_with_logits(
                attr_logits, attr_labels.float()
            )
        
        # 5. Regularization loss (reference TransZero's compute_reg_loss)
        # TransZero method: 
        #   tgt = torch.matmul(in_package['batch_label'], self.att)  # Get target attribute vector
        #   embed = in_package['embed']  # Visual embedding in semantic space
        #   loss_reg = F.mse_loss(embed, tgt, reduction='mean')
        # 
        # Our implementation:
        #   - Use attr_embed (normalized attribute predictions) as the embedding
        #   - Use class_attr_map[labels] as the target attribute vector
        loss_reg = torch.tensor(0.0, device=labels.device)
        if class_attr_map is not None:
            # Get true class attribute vector for each sample (reference TransZero)
            target_attrs = class_attr_map[labels]  # (batch_size, num_attrs)
            
            # Use attr_embed if available (normalized attribute predictions), otherwise use sigmoid(attr_logits)
            if 'attr_embed' in outputs:
                attr_embed = outputs['attr_embed']  # (batch_size, num_attrs) - already normalized
            else:
                attr_embed = torch.sigmoid(attr_logits)  # (batch_size, num_attrs) - normalize to [0,1]
            
            # Constrain attribute predictions to match true class attribute vectors (reference TransZero)
            # Reference TransZero: loss_reg = F.mse_loss(embed, tgt, reduction='mean')
            # Our implementation: Use MSE loss (primary) + cosine similarity loss (auxiliary) for better alignment
            # Normalize both for cosine similarity
            attr_embed_norm = F.normalize(attr_embed, p=2, dim=-1)
            target_attrs_norm = F.normalize(target_attrs, p=2, dim=-1)
            
            # Primary loss: MSE for magnitude (reference TransZero)
            loss_reg_mse = F.mse_loss(attr_embed, target_attrs, reduction='mean')
            
            # Auxiliary loss: Cosine distance for direction (helps with alignment)
            loss_reg_cos = 1.0 - (attr_embed_norm * target_attrs_norm).sum(dim=-1).mean()  # Cosine distance
            
            # Combine both losses (MSE is primary, cosine is auxiliary)
            # Reduce regularization strength to prevent overfitting to training set attribute patterns
            # Strong regularization forces attributes to match class vectors exactly, losing generalization
            loss_reg = loss_reg_mse + 0.2 * loss_reg_cos  # Further reduced cosine weight to prevent overfitting
        
        loss_continuity = torch.tensor(0.0, device=labels.device)
        loss_kernel_reg = torch.tensor(0.0, device=labels.device)
        loss_threshold_reg = torch.tensor(0.0, device=labels.device)

        # Total loss (including attribute prediction loss)
        loss = (
            self.lambda_attr * loss_attr +
            self.lambda_cls * loss_cls +
            self.lambda_fus * loss_fus +
            self.lambda_reg * loss_reg
        )
        
        # If attribute labels are available, add attribute prediction loss
        if attr_labels is not None:
            loss = loss + self.lambda_attr_pred * loss_attr_pred
        
        return {
            'loss': loss,
            'loss_attr': loss_attr,
            'loss_cls': loss_cls,
            'loss_fus': loss_fus,
            'loss_attr_pred': loss_attr_pred,
            'loss_reg': loss_reg,
            'loss_continuity': loss_continuity,
            'loss_kernel_reg': loss_kernel_reg,
            'loss_threshold_reg': loss_threshold_reg,
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

