#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Breast ultrasound image dataset loader for 3-class classification (benign, malignant, normal)
Filters out mask images (files containing "_mask" in filename)
Supports train/test separation, no seen/unseen distinction (all visible classes)
"""
import os
import random
import sys
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split


class FetalImageDataset(Dataset):
    """Breast ultrasound image dataset (3 classes: benign, malignant, normal)"""
    
    def __init__(
        self,
        data_root: str,
        split: str,  # 'train', 'val' or 'test'
        class_keys: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None,
        samples_list: Optional[List[Tuple[str, int]]] = None,
    ):
        """
        Args:
            data_root: 数据根目录（包含 train / val / test 三个子目录）
            split: 数据集划分（'train'、'val' 或 'test'）
            class_keys: 类别列表（如果为 None，则从 split 对应目录自动检测）
            transform: 图像增广 / 预处理
            samples_list: 可选的样本列表，如果提供则直接使用，不从目录加载
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        
        # Auto-detect classes or use provided classes
        if class_keys is None:
            self.class_keys = self._detect_classes()
        else:
            self.class_keys = class_keys
        
        # Build class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_keys)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load all samples (either from samples_list or from directory)
        if samples_list is not None:
            self.samples = samples_list
        else:
            self.samples = self._load_samples()
        
        print(f"[BreastDataset] {split} dataset:")
        print(f"  - Number of classes: {len(self.class_keys)}")
        print(f"  - Number of samples: {len(self.samples)}")
        self._print_class_stats()
    
    def _detect_classes(self) -> List[str]:
        """Auto-detect classes from folders"""
        split_dir = os.path.join(self.data_root, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Data directory does not exist: {split_dir}")
        
        classes = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])
        
        if not classes:
            raise ValueError(f"No class folders found in {split_dir}")
        
        return classes
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """
        Load all sample paths and labels
        CRITICAL: Filter out mask images (files containing "_mask" in filename)
        Mask images should NOT be used for training - they are only for visualization/evaluation
        """
        samples = []
        split_dir = os.path.join(self.data_root, self.split)
        
        for class_name in self.class_keys:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Load all image files, but EXCLUDE mask images
            for filename in sorted(os.listdir(class_dir)):
                # Skip mask images - they contain "_mask" in filename
                if "_mask" in filename.lower():
                    continue
                
                # Only load image files
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, filename)
                    samples.append((img_path, class_idx))
        
        return samples
    
    def _print_class_stats(self):
        """Print sample count statistics for each class"""
        from collections import Counter
        class_counts = Counter([label for _, label in self.samples])
        
        print(f"  - Samples per class:")
        for class_idx, count in sorted(class_counts.items()):
            class_name = self.idx_to_class[class_idx]
            print(f"    {class_name}: {count}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, int, str]:
        """
        Returns:
            image: PIL Image
            label: int (class index)
            img_path: str (image path)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Unable to load image {img_path}: {e}")
            # Return a black image as placeholder
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transformation
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights (for handling class imbalance)"""
        from collections import Counter
        class_counts = Counter([label for _, label in self.samples])
        total = len(self.samples)
        
        weights = torch.zeros(len(self.class_keys))
        for class_idx in range(len(self.class_keys)):
            count = class_counts.get(class_idx, 1)
            weights[class_idx] = total / (len(self.class_keys) * count)
        
        return weights


def get_default_transform(image_size: int = 224, is_training: bool = True):
    """
    Get default image transformation for breast ultrasound images
    Common preprocessing for BUSI dataset:
    - Resize to 256x256
    - Random crop (training) or center crop (validation/test) to 224x224
    - Random horizontal flip (training only)
    - Normalize with ImageNet statistics
    """
    if is_training:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    
    return transform


def _seed_worker(worker_id: int, base_seed: int = 42):
    """固定每个 DataLoader worker 的随机种子，保证多跑可复现（含 torch，供 RandomCrop 等使用）。"""
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_fetal_dataloaders(
    data_root: str,
    class_keys: Optional[List[str]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    val_split_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    构建训练 / 验证 / 测试三套数据加载器。
    
    支持两种数据目录结构：
    1. 标准结构（包含val目录）：
        data_root/
            train/
            val/
            test/
    
    2. 新结构（只有train和test，需要从train中划分val）：
        data_root/
            train/
            test/
    
    为了避免测试集数据泄露：
      - 训练阶段只访问 train/ 和 val/ 目录（或从train划分的val）
      - test/ 目录只在最终测试阶段使用一次，绝不参与训练
    
    注意：自动过滤掉所有包含"_mask"的文件名，确保mask图片不参与训练
    """
    # 检查val目录是否存在
    val_dir = os.path.join(data_root, 'val')
    has_val_dir = os.path.isdir(val_dir)
    
    # 1) 训练集（train/）
    train_dataset_full = FetalImageDataset(
        data_root=data_root,
        split='train',
        class_keys=class_keys,
        transform=get_default_transform(image_size, is_training=True),
    )
    
    # 2) 验证集的处理
    if has_val_dir:
        # 如果val目录存在，使用独立的val目录（旧方式）
        print(f"\n[DataLoader] Using separate val/ directory")
        val_dataset = FetalImageDataset(
            data_root=data_root,
            split='val',
            class_keys=train_dataset_full.class_keys,  # 与 train 保持完全一致的类别顺序
            transform=get_default_transform(image_size, is_training=False),
        )
        train_dataset = train_dataset_full
    else:
        # 如果val目录不存在，从train中划分验证集（新方式）
        print(f"\n[DataLoader] val/ directory not found, splitting train/ into train and val")
        print(f"[DataLoader] Using {val_split_ratio*100:.1f}% of train data for validation")
        
        # 收集所有样本的索引和标签，用于分层划分
        all_indices = list(range(len(train_dataset_full)))
        all_labels = [label for _, label in train_dataset_full.samples]
        
        # 使用分层划分，确保每个类别都有适当比例
        train_indices, val_indices = train_test_split(
            all_indices,
            test_size=val_split_ratio,
            stratify=all_labels,
            random_state=42,  # 固定随机种子以确保可重现
        )
        
        # 创建划分后的样本列表
        train_samples = [train_dataset_full.samples[i] for i in train_indices]
        val_samples = [train_dataset_full.samples[i] for i in val_indices]
        
        # 创建新的数据集实例，使用划分后的样本和不同的transform
        train_dataset = FetalImageDataset(
            data_root=data_root,
            split='train',  # 虽然是从train划分的，但标记为train
            class_keys=train_dataset_full.class_keys,
            transform=get_default_transform(image_size, is_training=True),
            samples_list=train_samples,  # 使用划分后的样本
        )
        
        val_dataset = FetalImageDataset(
            data_root=data_root,
            split='val',  # 标记为val
            class_keys=train_dataset_full.class_keys,
            transform=get_default_transform(image_size, is_training=False),
            samples_list=val_samples,  # 使用划分后的样本
        )
        
        # 打印划分统计
        from collections import Counter
        train_labels = [label for _, label in train_samples]
        val_labels = [label for _, label in val_samples]
        train_counts = Counter(train_labels)
        val_counts = Counter(val_labels)
        print(f"[DataLoader] Train/Val split statistics:")
        for class_idx in sorted(train_counts.keys()):
            class_name = train_dataset_full.idx_to_class[class_idx]
            train_count = train_counts[class_idx]
            val_count = val_counts[class_idx]
            total_count = train_count + val_count
            print(f"  {class_name}: train={train_count}, val={val_count}, total={total_count}")
    
    # 3) 数据加载器（固定 generator 与 worker_init_fn 以保证可复现）
    g = torch.Generator()
    g.manual_seed(seed)

    def seed_worker(worker_id: int):
        _seed_worker(worker_id, base_seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    g_val = torch.Generator()
    g_val.manual_seed(seed)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g_val,
    )

    # 4) 测试集（test/）——仅在最终评估阶段使用，绝不参与训练
    test_dataset = FetalImageDataset(
        data_root=data_root,
        split='test',
        class_keys=train_dataset_full.class_keys,  # Use same classes
        transform=get_default_transform(image_size, is_training=False),
    )

    g_test = torch.Generator()
    g_test.manual_seed(seed)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g_test,
    )
    
    print(f"\n[DataLoader] Created data loaders:")
    if not has_val_dir:
        print(f"  - Training set (from train/, {100-val_split_ratio*100:.1f}%): {len(train_dataset)} samples")
        print(f"  - Validation set (from train/, {val_split_ratio*100:.1f}%): {len(val_dataset)} samples")
    else:
        print(f"  - Training set (train/): {len(train_dataset)} samples")
        print(f"  - Validation set (val/): {len(val_dataset)} samples")
    print(f"  - Test set (test/): {len(test_dataset)} samples")
    print(f"  - Test set is NOT used in training (data leakage prevention)")
    print(f"  - Mask images (containing '_mask') are automatically filtered out")
    
    return train_loader, val_loader, test_loader, train_dataset_full.class_keys


if __name__ == "__main__":
    # Test code
    data_root = "path/to/data"
    train_loader, val_loader, test_loader, class_keys = get_fetal_dataloaders(
        data_root=data_root,
        batch_size=32,
    )
    
    print(f"Class list: {class_keys}")
    
    # Test loading one batch
    for images, labels in train_loader:
        print(f"Image shape: {images.shape}")
        print(f"Label shape: {labels.shape}")
        break
