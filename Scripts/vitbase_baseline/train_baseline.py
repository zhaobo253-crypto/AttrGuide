#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train baseline breast ultrasound image classification model (ViT only, no attribute guidance)

This script:
1) Creates dataloaders for train/val/test splits
2) Trains the baseline ViT model with CrossEntropyLoss
"""
import argparse
import os
import random
import sys
from datetime import datetime
from typing import Dict, List

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 鍙鐜帮紝椤诲湪 import torch 鍓?
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import wandb

try:
    from data_utils.FetalDatasetSimple import get_fetal_dataloaders
except ImportError as e:
    print(f"[ERROR] Failed to import get_fetal_dataloaders: {e}")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "FetalDatasetSimple", 
        os.path.join(os.path.dirname(__file__), "data_utils", "FetalDatasetSimple.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load FetalDatasetSimple.py file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'get_fetal_dataloaders'):
        raise ImportError("get_fetal_dataloaders function not found in FetalDatasetSimple.py")
    get_fetal_dataloaders = module.get_fetal_dataloaders
    print(f"[INFO] Successfully loaded get_fetal_dataloaders from file")
from models.BaselineClassifier import BaselineClassifier


# -----------------------------
# Utility helpers
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def compute_classification_accuracy(logits, labels):
    """Compute classification accuracy"""
    preds = logits.argmax(dim=1)
    correct = (preds == labels).float()
    return correct.mean().item()


def compute_per_class_accuracy_from_predictions(preds, labels, num_classes):
    """Compute per-class accuracy from predictions"""
    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)
    
    for cls_idx in range(num_classes):
        mask = (labels == cls_idx)
        per_class_total[cls_idx] = mask.sum().item()
        if per_class_total[cls_idx] > 0:
            per_class_correct[cls_idx] = (preds[mask] == labels[mask]).sum().item()
    
    per_class_acc = {}
    for cls_idx in range(num_classes):
        if per_class_total[cls_idx] > 0:
            per_class_acc[cls_idx] = per_class_correct[cls_idx] / per_class_total[cls_idx]
        else:
            per_class_acc[cls_idx] = 0.0
    
    return per_class_acc


# -----------------------------
# Training / Validation / Testing
# -----------------------------

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    scaler=None,
):
    model.train()
    total_loss = 0.0
    correct_cls = 0.0

    for batch_idx, batch_data in enumerate(train_loader):
        if len(batch_data) == 2:
            images, labels = batch_data
        elif len(batch_data) == 3:
            images, labels, _ = batch_data
        else:
            raise ValueError(f"Unexpected batch_data length: {len(batch_data)}")
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs['cls_logits'], labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs['cls_logits'], labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        correct_cls += compute_classification_accuracy(outputs['cls_logits'], labels)

    n = len(train_loader)
    return {
        "loss": total_loss / n,
        "acc_cls": correct_cls / n,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()
    total_loss = 0.0
    correct_cls = 0.0

    for batch_data in loader:
        if len(batch_data) == 2:
            images, labels = batch_data
        elif len(batch_data) == 3:
            images, labels, _ = batch_data
        else:
            raise ValueError(f"Unexpected batch_data length: {len(batch_data)}")
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs['cls_logits'], labels)

        total_loss += loss.item()
        correct_cls += compute_classification_accuracy(outputs['cls_logits'], labels)

    n = len(loader)
    return {
        "loss": total_loss / n,
        "acc_cls": correct_cls / n,
    }


@torch.no_grad()
def test(model, loader, device, num_classes):
    model.eval()
    correct_cls = 0.0
    all_preds_cls, all_labels = [], []

    for batch_data in loader:
        if len(batch_data) == 2:
            images, labels = batch_data
        elif len(batch_data) == 3:
            images, labels, _ = batch_data
        else:
            raise ValueError(f"Unexpected batch_data length: {len(batch_data)}")
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)

        correct_cls += compute_classification_accuracy(outputs['cls_logits'], labels)
        all_preds_cls.append(outputs['cls_logits'].argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    n = len(loader)
    acc_cls = correct_cls / n

    all_preds_cls = np.concatenate(all_preds_cls)
    all_labels = np.concatenate(all_labels)

    per_class_cls = compute_per_class_accuracy_from_predictions(
        torch.from_numpy(all_preds_cls),
        torch.from_numpy(all_labels),
        num_classes,
    )

    report = classification_report(
        all_labels,
        all_preds_cls,
        digits=4,
        output_dict=False,
    )
    cm = confusion_matrix(all_labels, all_preds_cls)

    return {
        "acc_cls": acc_cls,
        "per_class_cls": per_class_cls,
        "report": report,
        "confusion_matrix": cm,
    }


# -----------------------------
# Main
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline Breast Ultrasound Image Classification Model (ViT only, 3 classes: benign, malignant, normal)")
    parser.add_argument("--data_root", type=str, default="./data/breastdata",
                        help="Dataset root containing train/ and test/ folders")
    parser.add_argument("--backbone", type=str, default="vitbase", choices=["vitbase"])
    parser.add_argument("--resnet_path", type=str, default="./checkpoints/pretrained/vit_b_16-c867db91.pth",
                        help="Path to pretrained backbone weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for classification head")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_baseline")
    parser.add_argument("--save_best_only", action="store_true", help="Only keep best model")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU")
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Use wandb for experiment tracking")
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false", help="Disable wandb")
    parser.add_argument("--wandb_project", type=str, default="AttrGuide", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name (auto-generated if not provided)")
    parser.add_argument("--wandb_offline", action="store_true", help="Force wandb to run in offline mode")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    set_seed(42)
    
    # Wandb setup
    if args.use_wandb:
        wandb_mode = "offline" if args.wandb_offline else "online"
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            mode=wandb_mode,
            config=vars(args),
        )
    
    print("\n==========================================")
    print("Train Baseline Breast Ultrasound Image Classification Model")
    print("ViT only, no attribute guidance")
    print("==========================================")
    print(f"Backbone: {args.backbone.upper()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"\nStart time: {datetime.now().strftime('%a %b %d %I:%M:%S %p %Z %Y')}")
    if args.use_wandb and wandb.run:
        print(f"Wandb run: {wandb.run.url if hasattr(wandb.run, 'url') else 'N/A'}")
    print("==========================================\n")
    print(f"Using device: {device}\n")

    # Load dataset
    print("=============== Loading Dataset ===============")
    train_loader, val_loader, test_loader, class_keys = get_fetal_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42,
    )
    
    num_classes = len(class_keys)
    print(f"\nFinal number of classes: {num_classes}")
    print(f"Final class list: {class_keys}\n")

    # Create model
    print("=============== Creating Model ===============")
    print(f"Backbone type: {args.backbone.upper()}")
    backbone_feat_dim = 768  # ViT-Base feature dimension
    print(f"Backbone feature dimension: {backbone_feat_dim} (ViT-Base)")
    
    # Check if BaselineClassifier supports dropout parameter
    import inspect
    sig = inspect.signature(BaselineClassifier.__init__)
    if 'dropout' in sig.parameters:
        model = BaselineClassifier(
            backbone_type=args.backbone,
            num_classes=num_classes,
            backbone_feat_dim=backbone_feat_dim,
            resnet_path=args.resnet_path,
            dropout=args.dropout,
        ).to(device)
    else:
        print(f"[WARNING] BaselineClassifier does not support dropout parameter, using default dropout=0.1")
        model = BaselineClassifier(
            backbone_type=args.backbone,
            num_classes=num_classes,
            backbone_feat_dim=backbone_feat_dim,
            resnet_path=args.resnet_path,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created successfully!")
    print(f"Total model parameters: {total_params:,}\n")

    # Optimizer / Loss / Scheduler / Mixed Precision
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.epochs // 3, gamma=0.5
    )
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    best_val_acc = -1.0

    print("=============== Starting Training ===============\n")
    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
        )
        val_stats = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(f"Epoch {epoch}/{args.epochs}:")
        print(
            f"  Train - Loss: {train_stats['loss']:.4f}, "
            f"ClsAcc: {train_stats['acc_cls']*100:.2f}%"
        )
        print(
            f"  Val   - Loss: {val_stats['loss']:.4f}, "
            f"ClsAcc: {val_stats['acc_cls']*100:.2f}%\n"
        )
        
        # Log metrics to wandb
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_stats['loss'],
                "train/acc_cls": train_stats['acc_cls'],
                "val/loss": val_stats['loss'],
                "val/acc_cls": val_stats['acc_cls'],
                "best_val_acc": best_val_acc,
                "learning_rate": scheduler.get_last_lr()[0],
            })

        # Save best model
        if val_stats["acc_cls"] > best_val_acc:
            best_val_acc = val_stats["acc_cls"]
            checkpoint = {
                'state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'epoch': epoch,
            }
            torch.save(checkpoint, best_model_path)
            print(f"  Saved best model to: {best_model_path} (Val Acc: {best_val_acc*100:.2f}%)\n")

        # Learning rate scheduling
        scheduler.step()

    print("\nTraining completed! Best validation accuracy: {:.2f}%".format(best_val_acc * 100))

    # Final testing
    print("\n=============== Final Testing ===============")
    if os.path.isfile(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded best model: {best_model_path} (Epoch {checkpoint.get('epoch', 'unknown')}, Val Acc: {checkpoint.get('best_val_acc', 0)*100:.2f}%)\n")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded best model: {best_model_path}\n")
    else:
        print("Warning: best model not found, using last epoch weights.\n")

    test_stats = test(
        model,
        test_loader,
        device,
        num_classes=num_classes,
    )

    import sys
    sys.stdout.flush()
    
    print("=============== Test Results ===============")
    print(f"Overall accuracy:")
    print(f"  ClsAcc:  {test_stats['acc_cls']*100:.2f}%\n")
    sys.stdout.flush()
    
    # Log test results to wandb
    if args.use_wandb and wandb.run:
        try:
            wandb.log({
                "test/acc_cls": test_stats['acc_cls'],
            })
            
            # Log per-class accuracies
            for idx in range(num_classes):
                class_name = class_keys[idx] if idx < len(class_keys) else str(idx)
                wandb.log({
                    f"test/per_class_cls/{class_name}": test_stats["per_class_cls"].get(idx, 0.0),
                })
        except Exception as e:
            print(f"[WARNING] Failed to log test results to wandb: {e}")

    print("Per-class accuracy (Direct classification):")
    for idx in range(num_classes):
        acc = test_stats["per_class_cls"].get(idx, 0.0) * 100
        name = class_keys[idx] if idx < len(class_keys) else str(idx)
        print(f"  {name}: {acc:.2f}%")
    print("")

    print("Classification Report (Direct predictions):")
    print(test_stats["report"])
    print("\nConfusion Matrix (Direct predictions):")
    print("Rows=true class, Columns=predicted class")
    print(f"Class order: {class_keys}")
    print(test_stats["confusion_matrix"])
    sys.stdout.flush()

    print("\n==========================================")
    print("Training completed!")
    print(f"End time: {datetime.now().strftime('%a %b %d %I:%M:%S %p %Z %Y')}")
    print("==========================================")
    
    if args.use_wandb and wandb.run:
        try:
            wandb.finish()
            print("[INFO] Wandb run finished.")
            sys.stdout.flush()
        except Exception as e:
            print(f"[WARNING] Error finishing wandb run: {str(e)}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

