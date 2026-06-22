#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train attribute-guided breast ultrasound image classification model (TransZero-style)

This script:
1) Loads attribute table and precomputed attribute embeddings
2) Builds class 鈫?folder mapping and the class-attribute matrix
3) Creates dataloaders for train/val/test splits
4) Trains the attribute-guided model with multi-branch losses
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
from sklearn.metrics import classification_report, confusion_matrix
import wandb

try:
    from data_utils.FetalDatasetSimple import get_fetal_dataloaders
except ImportError as e:
    print(f"[ERROR] Failed to import get_fetal_dataloaders: {e}")
    print(f"[INFO] Checking if file exists and contains the function...")
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
from models.FetalAttributeClassifier import FetalAttributeClassifier
from utils.attribute_processor import AttributeProcessor
from utils.losses import (
    AttributeGuidedLoss,
    compute_classification_accuracy,
    compute_per_class_accuracy_from_predictions,
)


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


def build_class_mapping(class_keys: List[str], attr_processor: AttributeProcessor, filter_unmatched: bool = True) -> Dict[str, str]:
    """
    灏嗘暟鎹泦涓殑绫诲埆鍚嶏紙鏂囦欢澶瑰悕锛夋槧灏勫埌灞炴€ц〃涓殑 key銆?    
    鏂扮増灞炴€ц〃锛堝睘鎬у搴?csv锛変腑宸茬粡鎻愪緵浜嗚嫳鏂囧垪 `name`锛屼笖涓庢暟鎹泦绫诲埆鍚嶄竴鑷达紝
    鍦?utils/attribute_processor.py 涓垜浠凡缁忚 attr_dict 浣跨敤 `name` 浣滀负 key锛?    鍥犳杩欓噷鍙互鐩存帴鍋氫竴涓€瀵瑰簲锛?
        class_keys = ['benign', 'malignant', 'normal']
        attr_processor.attr_dict.keys() = ['benign', 'malignant', 'normal']

    Args:
        class_keys: 鏁版嵁闆嗕腑鐨勬墍鏈夌被鍒?        attr_processor: 灞炴€у鐞嗗櫒
        filter_unmatched: 濡傛灉涓篢rue锛屽彧杩斿洖鍖归厤鐨勭被鍒紱濡傛灉涓篎alse锛屼笉鍖归厤鐨勭被鍒細鎶ラ敊
    
    Returns:
        mapping: {class_key: attribute_key} 瀛楀吀锛屽彧鍖呭惈鍖归厤鐨勭被鍒?    """
    # 鎵撳嵃灞炴€ц〃涓彲鐢ㄧ殑绫诲埆 key锛堢幇鍦ㄦ槸鑻辨枃鍚嶏紝濡?benign, malignant, normal 绛夛級
    print("\nKeys in attribute CSV (total {}):".format(len(attr_processor.attr_dict)))
    for idx, k in enumerate(attr_processor.attr_dict.keys(), start=1):
        print(f"  {idx:2d}. {k}")

    mapping: Dict[str, str] = {}
    missing_classes: List[str] = []

    print("\nBuilding class-to-attribute mapping (total {} classes in dataset)...".format(len(class_keys)))
    for ck in class_keys:
        if ck in attr_processor.attr_dict:
            # 鐩存帴鐢ㄧ浉鍚岀殑 key锛歝lass_key -> class_key
            mapping[ck] = ck
            print(f"  [OK] '{ck}' -> '{ck}'")
        else:
            missing_classes.append(ck)
            print(f"  [SKIP] Class '{ck}' not found in attribute CSV keys (will be filtered out)")

    if missing_classes:
        available = list(attr_processor.attr_dict.keys())
        if filter_unmatched:
            print(f"\n[AttributeProcessor] Warning: {len(missing_classes)} dataset classes were not found in the attribute table and will be filtered out.")
            for ck in missing_classes:
                print(f"  - {ck}")
            print(f"\n[AttributeProcessor] Training will use only the {len(mapping)} matched classes.")
            print(f"Available attribute keys: {available}\n")
        else:
            error_msg = "\n[AttributeProcessor] The following dataset classes were not found in the attribute table:\n"
            for ck in missing_classes:
                error_msg += f"  - {ck}\n"
            error_msg += "\nPlease ensure that the CSV contains a 'name' or 'folder' column matching the dataset class folders.\n"
            error_msg += f"\nAvailable attribute keys:\n  {available}\n"
            raise ValueError(error_msg)

    print("\n============================================================")
    print("Final Mapping Relationship (class_key -> attribute_key):")
    print(f"Total matched classes: {len(mapping)}")
    print("============================================================")
    for ck, key in mapping.items():
        print(f"  [OK] {ck:<15} -> {key}")
    print("============================================================")
    return mapping


# -----------------------------
# Training / Validation / Testing
# -----------------------------

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    attr_embeddings,
    class_attr_map,
    scaler=None,  # Mixed precision scaler
):
    model.train()
    total_loss = 0.0
    total_attr = total_cls = total_fus = total_reg = total_attr_pred = total_continuity = total_kernel_reg = total_threshold_reg = 0.0
    correct_attr = correct_cls = correct_fus = 0.0

    for batch_idx, batch_data in enumerate(train_loader):
        # Handle different return formats: (images, labels) or (images, labels, paths)
        if len(batch_data) == 2:
            images, labels = batch_data
        elif len(batch_data) == 3:
            images, labels, _ = batch_data  # Ignore img_path for training
        else:
            raise ValueError(f"Unexpected batch_data length: {len(batch_data)}")
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # Ensure attr_embeddings has the same dtype as images in autocast context
                attr_emb = attr_embeddings.to(images.dtype)
                outputs = model(images, attr_emb)
                attr_labels = class_attr_map[labels]
                loss_dict = criterion(
                    outputs,
                    labels,
                    attr_labels=attr_labels,
                    class_attr_map=class_attr_map,
                )
                loss = loss_dict["loss"]
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Ensure attr_embeddings has the same dtype as images
            attr_emb = attr_embeddings.to(images.dtype)
            outputs = model(images, attr_emb)
            attr_labels = class_attr_map[labels]
            loss_dict = criterion(
                outputs,
                labels,
                attr_labels=attr_labels,
                class_attr_map=class_attr_map,
            )
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

        # accumulate
        total_loss += loss.item()
        total_attr += loss_dict["loss_attr"].item()
        total_cls += loss_dict["loss_cls"].item()
        total_fus += loss_dict["loss_fus"].item()
        total_reg += loss_dict["loss_reg"].item()
        total_attr_pred += loss_dict["loss_attr_pred"].item()
        total_continuity += loss_dict.get("loss_continuity", 0.0)
        total_kernel_reg += loss_dict.get("loss_kernel_reg", 0.0)
        total_threshold_reg += loss_dict.get("loss_threshold_reg", 0.0)

        correct_attr += compute_classification_accuracy(outputs["attr_cls_logits"], labels)
        correct_cls += compute_classification_accuracy(outputs["cls_logits"], labels)
        correct_fus += compute_classification_accuracy(outputs["fus_logits"], labels)

    n = len(train_loader)
    return {
        "loss": total_loss / n,
        "loss_attr": total_attr / n,
        "loss_cls": total_cls / n,
        "loss_fus": total_fus / n,
        "loss_reg": total_reg / n,
        "loss_attr_pred": total_attr_pred / n,
        "loss_continuity": total_continuity / n,
        "loss_kernel_reg": total_kernel_reg / n,
        "loss_threshold_reg": total_threshold_reg / n,
        "acc_attr": correct_attr / n,
        "acc_cls": correct_cls / n,
        "acc_fus": correct_fus / n,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    attr_embeddings,
    class_attr_map,
):
    model.eval()
    total_loss = 0.0
    total_attr = total_cls = total_fus = total_reg = total_attr_pred = total_continuity = total_kernel_reg = total_threshold_reg = 0.0
    correct_attr = correct_cls = correct_fus = 0.0

    for batch_data in loader:
        # Handle different return formats: (images, labels) or (images, labels, paths)
        if len(batch_data) == 2:
            images, labels = batch_data
        elif len(batch_data) == 3:
            images, labels, _ = batch_data  # Ignore img_path for validation
        else:
            raise ValueError(f"Unexpected batch_data length: {len(batch_data)}")
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Ensure attr_embeddings has the same dtype as model parameters
        # This is important for mixed precision training compatibility
        # Convert to float32 to avoid dtype mismatch issues in evaluation
        attr_emb = attr_embeddings.to(images.dtype)
        
        outputs = model(images, attr_emb)
        attr_labels = class_attr_map[labels]

        loss_dict = criterion(
            outputs,
            labels,
            attr_labels=attr_labels,
            class_attr_map=class_attr_map,
        )

        total_loss += loss_dict["loss"].item()
        total_attr += loss_dict["loss_attr"].item()
        total_cls += loss_dict["loss_cls"].item()
        total_fus += loss_dict["loss_fus"].item()
        total_reg += loss_dict["loss_reg"].item()
        total_attr_pred += loss_dict["loss_attr_pred"].item()
        total_continuity += loss_dict.get("loss_continuity", 0.0)
        total_kernel_reg += loss_dict.get("loss_kernel_reg", 0.0)
        total_threshold_reg += loss_dict.get("loss_threshold_reg", 0.0)

        correct_attr += compute_classification_accuracy(outputs["attr_cls_logits"], labels)
        correct_cls += compute_classification_accuracy(outputs["cls_logits"], labels)
        correct_fus += compute_classification_accuracy(outputs["fus_logits"], labels)

    n = len(loader)
    return {
        "loss": total_loss / n,
        "loss_attr": total_attr / n,
        "loss_cls": total_cls / n,
        "loss_fus": total_fus / n,
        "loss_reg": total_reg / n,
        "loss_attr_pred": total_attr_pred / n,
        "loss_continuity": total_continuity / n,
        "loss_kernel_reg": total_kernel_reg / n,
        "loss_threshold_reg": total_threshold_reg / n,
        "acc_attr": correct_attr / n,
        "acc_cls": correct_cls / n,
        "acc_fus": correct_fus / n,
    }


@torch.no_grad()
def test(model, loader, device, attr_embeddings, class_attr_map, num_classes):
    model.eval()
    correct_attr = correct_cls = correct_fus = 0.0
    all_preds_attr, all_preds_cls, all_preds_fus, all_labels = [], [], [], []

    for batch_data in loader:
        # Handle different return formats: (images, labels) or (images, labels, paths)
        if len(batch_data) == 2:
            images, labels = batch_data
        elif len(batch_data) == 3:
            images, labels, _ = batch_data  # Ignore img_path for validation
        else:
            raise ValueError(f"Unexpected batch_data length: {len(batch_data)}")
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Ensure attr_embeddings has the same dtype as model parameters
        attr_emb = attr_embeddings.to(images.dtype)
        
        outputs = model(images, attr_emb)

        correct_attr += compute_classification_accuracy(outputs["attr_cls_logits"], labels)
        correct_cls += compute_classification_accuracy(outputs["cls_logits"], labels)
        correct_fus += compute_classification_accuracy(outputs["fus_logits"], labels)

        all_preds_attr.append(outputs["attr_cls_logits"].argmax(dim=1).cpu().numpy())
        all_preds_cls.append(outputs["cls_logits"].argmax(dim=1).cpu().numpy())
        all_preds_fus.append(outputs["fus_logits"].argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    n = len(loader)
    acc_attr = correct_attr / n
    acc_cls = correct_cls / n
    acc_fus = correct_fus / n

    all_preds_attr = np.concatenate(all_preds_attr)
    all_preds_cls = np.concatenate(all_preds_cls)
    all_preds_fus = np.concatenate(all_preds_fus)
    all_labels = np.concatenate(all_labels)

    per_class_attr = compute_per_class_accuracy_from_predictions(
        torch.from_numpy(all_preds_attr),
        torch.from_numpy(all_labels),
        num_classes,
    )
    per_class_cls = compute_per_class_accuracy_from_predictions(
        torch.from_numpy(all_preds_cls),
        torch.from_numpy(all_labels),
        num_classes,
    )
    per_class_fus = compute_per_class_accuracy_from_predictions(
        torch.from_numpy(all_preds_fus),
        torch.from_numpy(all_labels),
        num_classes,
    )

    report = classification_report(
        all_labels,
        all_preds_fus,
        digits=4,
        output_dict=False,
    )
    cm = confusion_matrix(all_labels, all_preds_fus)

    return {
        "acc_attr": acc_attr,
        "acc_cls": acc_cls,
        "acc_fus": acc_fus,
        "per_class_attr": per_class_attr,
        "per_class_cls": per_class_cls,
        "per_class_fus": per_class_fus,
        "report": report,
        "confusion_matrix": cm,
    }


# -----------------------------
# Main
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train Attribute-Guided Breast Ultrasound Image Classification Model (3 classes: benign, malignant, normal)")
    parser.add_argument("--data_root", type=str, default="./data/breastdata",
                        help="Dataset root containing train/ and test/ folders")
    parser.add_argument("--attr_csv", type=str, default="./data/attributes_breast.csv",
                        help="CSV file with attribute table (default: /scratch/liyapeng/zhaobo/鍏紑涔宠吅3绫诲睘鎬?csv for 3-class breast dataset)")
    parser.add_argument("--precomputed_attr_emb", type=str, default="./data/attribute_embeddings_3cls_breast.pt",
                        help="Path to precomputed attribute embeddings (.pt/.pth/.npy)")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet101", "vitbase"])
    parser.add_argument("--resnet_path", type=str, default="./checkpoints/pretrained/resnet50-11ad3fa6.pth",
                        help="Path to pretrained backbone weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for classification head")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fusion_weight", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lambda_attr", type=float, default=0.5)
    parser.add_argument("--lambda_cls", type=float, default=0.3)
    parser.add_argument("--lambda_fus", type=float, default=0.2)
    parser.add_argument("--lambda_reg", type=float, default=0.5)
    parser.add_argument("--lambda_attr_pred", type=float, default=0.1)
    parser.add_argument("--use_scheduler", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_best_only", action="store_true", help="Only keep best model")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU")
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Use wandb for experiment tracking")
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false", help="Disable wandb")
    parser.add_argument("--wandb_project", type=str, default="AttrGuide", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name (auto-generated if not provided)")
    parser.add_argument("--wandb_offline", action="store_true", help="Force wandb to run in offline mode")
    return parser.parse_args()


def sanitize_string_for_wandb(s):
    """
    Convert a string to ASCII-safe representation for wandb.
    Uses 'replace' error handling to replace non-ASCII characters with '?'.
    
    Args:
        s: String that may contain non-ASCII characters
        
    Returns:
        ASCII-safe string
    """
    if not isinstance(s, str):
        return s
    try:
        # Try to encode as ASCII - if it works, return as-is
        s.encode('ascii')
        return s
    except UnicodeEncodeError:
        # Replace non-ASCII characters with '?'
        return s.encode('ascii', errors='replace').decode('ascii')


def sanitize_config_for_wandb(config_dict):
    """
    Clean config dictionary to remove non-ASCII characters that cause wandb encoding errors.
    
    Args:
        config_dict: Dictionary of configuration parameters
        
    Returns:
        Cleaned dictionary with safe ASCII-only strings
    """
    import os
    
    def sanitize_value(v):
        """Convert non-ASCII strings to safe ASCII representation"""
        if isinstance(v, str):
            # Use basename for file paths to avoid full path issues
            if os.path.exists(v) or '/' in v or '\\' in v:
                # It's a path, try to use basename, but sanitize it too
                basename = os.path.basename(v)
                if basename:
                    return sanitize_string_for_wandb(basename)
                else:
                    return sanitize_string_for_wandb(v)
            else:
                # Not a path, sanitize the string directly
                return sanitize_string_for_wandb(v)
        elif isinstance(v, (list, tuple)):
            return [sanitize_value(item) for item in v]
        elif isinstance(v, dict):
            return {sanitize_string_for_wandb(str(k)): sanitize_value(val) for k, val in v.items()}
        else:
            # For other types, convert to string and sanitize if needed
            try:
                str_v = str(v)
                if isinstance(v, (int, float, bool, type(None))):
                    return v  # Keep numeric and boolean types as-is
                else:
                    return sanitize_string_for_wandb(str_v)
            except:
                return v
    
    cleaned_config = {}
    for key, value in config_dict.items():
        try:
            # Sanitize both key and value
            safe_key = sanitize_string_for_wandb(str(key))
            cleaned_config[safe_key] = sanitize_value(value)
        except Exception as e:
            # If cleaning fails, skip this key or use a placeholder
            print(f"[WARNING] Failed to sanitize config key '{key}': {e}, skipping")
            # Skip problematic keys instead of using placeholder
    
    return cleaned_config


def main():
    args = parse_args()

    # 寮哄埗浣跨敤 ViT-Base 鍙婂搴旈璁粌鏉冮噸锛岄伩鍏嶈剼鏈弬鏁?鎹㈣瀵艰嚧鐨勯骞蹭笉涓€鑷?    args.backbone = "vitbase"
    if args.resnet_path is None or "vit_b_16" not in args.resnet_path:
        args.resnet_path = "./checkpoints/pretrained/vit_b_16-c867db91.pth"

    set_seed(42)

    device = "cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu"
    
    # Initialize wandb - automatically save code and track experiments
    # This will save all code files in the current directory to wandb
    wandb_run = None
    if args.use_wandb:
        run_name = args.wandb_name if args.wandb_name else f"lambda_reg_{args.lambda_reg}_attr_{args.lambda_attr}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Sanitize run_name to ensure it's ASCII-safe
        run_name = sanitize_string_for_wandb(run_name)
        
        # Determine mode: offline if explicitly requested or if network issues
        wandb_mode = "offline" if args.wandb_offline else "online"
        
        # Sanitize config to avoid Unicode encoding errors with wandb
        wandb_config = sanitize_config_for_wandb(vars(args))
        
        # Sanitize project name and tags
        wandb_project = sanitize_string_for_wandb(args.wandb_project)
        wandb_tags = [sanitize_string_for_wandb(tag) for tag in ["attribute-guided", "breast-ultrasound", f"lambda_reg_{args.lambda_reg}"]]
        
        if args.wandb_offline:
            print("\n[INFO] Wandb running in offline mode (--wandb_offline specified)")
            print("[INFO] Code and metrics will be saved locally.")
            print("[INFO] To sync later, run: wandb sync <run_directory>\n")
            try:
                wandb_run = wandb.init(
                    project=wandb_project,
                    name=run_name,
                    config=wandb_config,
                    save_code=False,  # Disable save_code to avoid path encoding issues
                    tags=wandb_tags,
                    mode="offline",
                )
            except (UnicodeEncodeError, UnicodeDecodeError) as e:
                print(f"\n[ERROR] Wandb initialization failed due to Unicode encoding error: {e}")
                print("[INFO] Disabling wandb and continuing training without logging.")
                args.use_wandb = False
                wandb_run = None
            except Exception as e:
                print(f"\n[WARNING] Wandb initialization failed: {e}")
                print("[INFO] Disabling wandb and continuing training without logging.")
                args.use_wandb = False
                wandb_run = None
        
        # Manually copy all code files to wandb code directory (create snapshot)
        if wandb_run is not None:
            import os
            import glob
            import shutil
            
            # Get wandb run directory
            if wandb_run and hasattr(wandb_run, 'dir'):
                run_dir = wandb_run.dir
                # Check if run_dir already contains 'files' directory
                if run_dir.endswith('files'):
                    code_dir = os.path.join(run_dir, "code")
                else:
                    code_dir = os.path.join(run_dir, "files", "code")
                os.makedirs(code_dir, exist_ok=True)
                
                code_dirs = ['models', 'utils', 'data_utils']
                code_files = ['train_fetal_attribute_model.py']
                # Also save SLURM script and other config files
                config_files = []
                # Don't save output files here - they will be saved at the end of training
                
                # Copy main training script
                for code_file in code_files:
                    if os.path.exists(code_file):
                        dest_path = os.path.join(code_dir, code_file)
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        shutil.copy2(code_file, dest_path)
                        print(f"[INFO] Saved code snapshot: {code_file}")
                
                # Copy config files (SLURM script, etc.)
                for config_file in config_files:
                    # Try current directory first
                    file_path = config_file
                    if not os.path.exists(file_path):
                        # Try with absolute path from current working directory
                        file_path = os.path.join(os.getcwd(), config_file)
                    if os.path.exists(file_path):
                        dest_path = os.path.join(code_dir, config_file)
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        shutil.copy2(file_path, dest_path)
                        print(f"[INFO] Saved config snapshot: {config_file}")
                    else:
                        print(f"[WARNING] Config file not found: {config_file} (checked: {file_path}, cwd: {os.getcwd()})")
                
                # Copy code from subdirectories (preserve directory structure)
                for code_dir_name in code_dirs:
                    if os.path.exists(code_dir_name):
                        py_files = glob.glob(os.path.join(code_dir_name, "**/*.py"), recursive=True)
                        for py_file in py_files:
                            # Preserve relative path structure
                            rel_path = py_file
                            dest_path = os.path.join(code_dir, rel_path)
                            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                            shutil.copy2(py_file, dest_path)
                            print(f"[INFO] Saved code snapshot: {py_file}")
                
                print(f"\n[INFO] Run directory: {run_dir}")
                print(f"[INFO] All code snapshots saved to: {code_dir}/\n")
            else:
                print("[WARNING] Could not determine wandb run directory, code snapshots may not be saved correctly.\n")
        else:
            try:
                # Try to initialize wandb with increased timeout
                wandb_run = wandb.init(
                    project=wandb_project,
                    name=run_name,
                    config=wandb_config,  # Use sanitized config
                    save_code=False,  # Disable save_code to avoid path encoding issues
                    tags=wandb_tags,
                    settings=wandb.Settings(
                        init_timeout=180,  # Increase timeout to 180 seconds
                        _disable_stats=False,
                    ),
                    mode="online",
                )
                # Manually copy all code files to wandb code directory (create snapshot)
                import os
                import glob
                import shutil
                
                # Get wandb run directory
                if wandb_run and hasattr(wandb_run, 'dir'):
                    run_dir = wandb_run.dir
                    # Check if run_dir already contains 'files' directory
                    if run_dir.endswith('files'):
                        code_dir = os.path.join(run_dir, "code")
                    else:
                        code_dir = os.path.join(run_dir, "files", "code")
                    os.makedirs(code_dir, exist_ok=True)
                    
                    code_dirs = ['models', 'utils', 'data_utils']
                    code_files = ['train_fetal_attribute_model.py']
                    # Also save SLURM script and other config files
                    config_files = []
                    
                    # Copy main training script
                    for code_file in code_files:
                        if os.path.exists(code_file):
                            dest_path = os.path.join(code_dir, code_file)
                            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                            shutil.copy2(code_file, dest_path)
                            print(f"[INFO] Saved code snapshot: {code_file}")
                    
                    # Copy config files (SLURM script, etc.)
                    for config_file in config_files:
                        if os.path.exists(config_file):
                            dest_path = os.path.join(code_dir, config_file)
                            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                            shutil.copy2(config_file, dest_path)
                            print(f"[INFO] Saved config snapshot: {config_file}")
                    
                    # Copy code from subdirectories (preserve directory structure)
                    for code_dir_name in code_dirs:
                        if os.path.exists(code_dir_name):
                            py_files = glob.glob(os.path.join(code_dir_name, "**/*.py"), recursive=True)
                            for py_file in py_files:
                                # Preserve relative path structure
                                rel_path = py_file
                                dest_path = os.path.join(code_dir, rel_path)
                                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                                shutil.copy2(py_file, dest_path)
                                print(f"[INFO] Saved code snapshot: {py_file}")
                    
                    print(f"[INFO] All code snapshots saved to: {code_dir}/\n")
            except Exception as e:
                print(f"\n[WARNING] Wandb online mode failed: {e}")
                print("[INFO] Attempting to switch to offline mode...")
                try:
                    # Fallback to offline mode
                    wandb_run = wandb.init(
                        project=wandb_project,
                        name=run_name,
                        config=wandb_config,  # Use sanitized config
                        save_code=False,  # Disable save_code to avoid path encoding issues
                        tags=wandb_tags,
                        mode="offline",
                    )
                    print("[INFO] Successfully switched to offline mode.")
                    print("[INFO] Code and metrics will be saved locally.")
                    print("[INFO] To sync later, run: wandb sync <run_directory>\n")
                except (UnicodeEncodeError, UnicodeDecodeError) as e2:
                    print(f"\n[ERROR] Wandb offline mode also failed due to Unicode encoding error: {e2}")
                    print("[INFO] Disabling wandb completely and continuing training without logging.")
                    args.use_wandb = False
                    wandb_run = None
                except Exception as e2:
                    print(f"\n[WARNING] Wandb offline mode also failed: {e2}")
                    print("[INFO] Disabling wandb completely and continuing training without logging.")
                    args.use_wandb = False
                    wandb_run = None
                
                # Manually copy all code files to wandb code directory (create snapshot)
                if wandb_run is not None:
                    import os
                    import glob
                    import shutil
                    
                    # Get wandb run directory
                    if wandb_run and hasattr(wandb_run, 'dir'):
                        run_dir = wandb_run.dir
                        # Check if run_dir already contains 'files' directory
                        if run_dir.endswith('files'):
                            code_dir = os.path.join(run_dir, "code")
                        else:
                            code_dir = os.path.join(run_dir, "files", "code")
                        os.makedirs(code_dir, exist_ok=True)
                        
                        code_dirs = ['models', 'utils', 'data_utils']
                        code_files = ['train_fetal_attribute_model.py']
                        # Also save SLURM script and other config files
                        config_files = []
                        
                        # Copy main training script
                        for code_file in code_files:
                            if os.path.exists(code_file):
                                dest_path = os.path.join(code_dir, code_file)
                                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                                shutil.copy2(code_file, dest_path)
                                print(f"[INFO] Saved code snapshot: {code_file}")
                        
                        # Copy config files (SLURM script, etc.)
                        for config_file in config_files:
                            if os.path.exists(config_file):
                                dest_path = os.path.join(code_dir, config_file)
                                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                                shutil.copy2(config_file, dest_path)
                                print(f"[INFO] Saved config snapshot: {config_file}")
                        
                        # Copy code from subdirectories (preserve directory structure)
                        for code_dir_name in code_dirs:
                            if os.path.exists(code_dir_name):
                                py_files = glob.glob(os.path.join(code_dir_name, "**/*.py"), recursive=True)
                                for py_file in py_files:
                                    # Preserve relative path structure
                                    rel_path = py_file
                                    dest_path = os.path.join(code_dir, rel_path)
                                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                                    shutil.copy2(py_file, dest_path)
                                    print(f"[INFO] Saved code snapshot: {py_file}")
                        
                        print(f"[INFO] All code snapshots saved to: {code_dir}/\n")
    
    print("\n==========================================")
    print("Train Attribute-Guided Breast Ultrasound Image Classification Model")
    print("Based on TransZero logic")
    print("==========================================")
    print(f"Backbone: {args.backbone.upper()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"\nStart time: {datetime.now().strftime('%a %b %d %I:%M:%S %p %Z %Y')}")
    if args.use_wandb and wandb.run:
        print(f"Wandb run: {wandb.run.url}")
    print("==========================================\n")
    print(f"Using device: {device}\n")

    # 1) Load attributes
    print("=============== Loading Attribute Information ===============")
    attr_processor = AttributeProcessor(args.attr_csv, use_clip=False)
    attr_info = attr_processor.get_attribute_info()
    num_attrs = attr_info["num_attrs"]

    # Load (or compute) attribute embeddings
    print(f"\n[INFO] Loading attribute embeddings from: {args.precomputed_attr_emb}")
    if os.path.exists(args.precomputed_attr_emb):
        print(f"[INFO] 鉁?Precomputed embeddings file exists")
        file_size = os.path.getsize(args.precomputed_attr_emb)
        print(f"[INFO] File size: {file_size} bytes ({file_size / 1024:.2f} KB)")
    else:
        print(f"[WARNING] 鈿?Precomputed embeddings file NOT found: {args.precomputed_attr_emb}")
        print(f"[WARNING] Will use random initialization (NOT recommended for training!)")
        print(f"[WARNING] Please generate embeddings first using: generate_attribute_embeddings.py")
    
    attr_embeddings = attr_processor.encode_attributes_with_clip(
        precomputed_path=args.precomputed_attr_emb
    )
    attr_embeddings = attr_embeddings.to(device)
    
    # CRITICAL: Store a hash/summary of embeddings to verify they're being used
    # This will help us verify in forward pass that the same embeddings are used
    attr_embeddings_sum = attr_embeddings.sum().item()
    attr_embeddings_first_few = attr_embeddings[0, :5].clone().detach().cpu().tolist()
    print(f"\n[VERIFICATION] Attribute embeddings summary:")
    print(f"  Total sum: {attr_embeddings_sum:.4f}")
    print(f"  First 5 values of first attribute: {[f'{x:.4f}' for x in attr_embeddings_first_few]}")
    print(f"  [INFO] These values will be used in model forward pass")
    print(f"  [INFO] Embeddings shape: {attr_embeddings.shape} -> will be passed as attrs_matrix to model.forward()")
    
    # Verify embedding source
    print(f"\n[INFO] Attribute embedding verification:")
    print(f"  Shape: {tuple(attr_embeddings.shape)}")
    if os.path.exists(args.precomputed_attr_emb):
        # Check if embeddings look like CLIP embeddings
        # CLIP embeddings are typically L2-normalized, so each vector has norm=1
        embedding_mean = attr_embeddings.mean().item()
        embedding_std = attr_embeddings.std().item()
        embedding_norms = torch.norm(attr_embeddings, dim=1)  # L2 norm of each embedding vector
        avg_norm = embedding_norms.mean().item()
        norm_std = embedding_norms.std().item()
        
        print(f"  Statistics:")
        print(f"    Mean: {embedding_mean:.4f}, Std: {embedding_std:.4f}")
        print(f"    Average L2 norm per vector: {avg_norm:.4f} (std: {norm_std:.4f})")
        
        # CLIP embeddings are normalized, so norm should be close to 1.0
        # Random embeddings typically have norm ~sqrt(512) 鈮?22.6 for std=1.0
        if 0.9 < avg_norm < 1.1 and norm_std < 0.1:
            print(f"  鉁?Embeddings are L2-normalized (typical of CLIP embeddings)")
            print(f"  鉁?Confirmed: Using CLIP-generated embeddings")
        elif avg_norm > 10:
            print(f"  鈿?WARNING: Embeddings have high norm ({avg_norm:.2f}), may be random initialization")
            print(f"  鈿?Expected normalized CLIP embeddings should have norm 鈮?1.0")
        else:
            print(f"  ? Embeddings loaded, but normalization status unclear")
            print(f"  ? If these are CLIP embeddings, they should be normalized (norm 鈮?1.0)")
    else:
        print(f"  鈿?Using random initialization (file not found)")
    print()

    # 2) Load dataset (first get all classes to check matching)
    print("=============== Loading Dataset ===============")
    # First load to get all available classes
    train_loader_all, _, _, class_keys_all = get_fetal_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42,
    )
    
    # 3) Build class mapping & filter to only matched classes
    print("\n=============== Building Class-Attribute Mapping ===============")
    class_to_folder = build_class_mapping(class_keys_all, attr_processor, filter_unmatched=True)
    
    # Get only matched class keys
    matched_class_keys = list(class_to_folder.keys())
    print(f"\n[INFO] Using {len(matched_class_keys)} matched classes out of {len(class_keys_all)} total classes")
    print(f"Matched classes: {matched_class_keys}")
    
    if len(matched_class_keys) < len(class_keys_all):
        print(f"\n[INFO] Filtering dataset to use only matched classes...")
        # Reload dataset with filtered classes
        train_loader, val_loader, test_loader, class_keys = get_fetal_dataloaders(
            data_root=args.data_root,
            class_keys=matched_class_keys,  # Only use matched classes
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=42,
        )
    else:
        # All classes matched, use original loaders
        train_loader = train_loader_all
        _, val_loader, test_loader, class_keys = get_fetal_dataloaders(
            data_root=args.data_root,
            class_keys=matched_class_keys,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=42,
        )
    
    num_classes = len(class_keys)
    print(f"\nFinal number of classes: {num_classes}")
    print(f"Final class list: {class_keys}\n")

    # 4) Build class-attribute matrix
    class_attr_map = attr_processor.build_class_attr_map(class_to_folder).to(device)
    print(f"\nClass-attribute mapping matrix shape: {tuple(class_attr_map.shape)}\n")

    # 5) Create model
    print("=============== Creating Model ===============")
    print(f"Backbone type: {args.backbone.upper()}")
    # Set backbone feature dimension based on backbone type
    if args.backbone == 'vitbase':
        backbone_feat_dim = 768  # ViT-Base feature dimension
        print(f"Backbone feature dimension: {backbone_feat_dim} (ViT-Base)")
    else:
        backbone_feat_dim = 2048  # ResNet50/101 feature dimension
        print(f"Backbone feature dimension: {backbone_feat_dim} (ResNet)")
    
    model = FetalAttributeClassifier(
        backbone_type=args.backbone,
        num_classes=num_classes,
        num_attrs=num_attrs,
        attr_emb_dim=attr_embeddings.shape[1],
        backbone_feat_dim=backbone_feat_dim,
        fusion_weight=args.fusion_weight,
        resnet_path=args.resnet_path,
        temperature=args.temperature,
        dropout=args.dropout,
    ).to(device)

    # Replace model's buffer with the fixed mapping (avoid training drift)
    # CRITICAL: Ensure class_attr_map is fixed and not trainable
    with torch.no_grad():
        model.class_attr_map.copy_(class_attr_map)
        model.class_attr_map.requires_grad_(False)  # Ensure it's not trainable

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created successfully!")
    print(f"Total model parameters: {total_params:,}\n")

    # 5) Optimizer / Loss / Scheduler / Mixed Precision
    # Reference TransZero: use SGD with momentum=0, weight_decay=0.0001
    # But Adam usually works better for our case, so keep Adam but adjust weight_decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,  # Weight decay for regularization
        betas=(0.9, 0.999),
    )
    # Use step LR scheduler instead of cosine (more stable)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.epochs // 3, gamma=0.5
    )
    # Mixed precision scaler for faster training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    criterion = AttributeGuidedLoss(
        lambda_attr=args.lambda_attr,
        lambda_cls=args.lambda_cls,
        lambda_fus=args.lambda_fus,
        lambda_reg=args.lambda_reg,
        lambda_attr_pred=args.lambda_attr_pred,
    )

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
            attr_embeddings,
            class_attr_map,
            scaler=scaler,  # Pass scaler for mixed precision
        )
        val_stats = evaluate(
            model,
            val_loader,
            criterion,
            device,
            attr_embeddings,
            class_attr_map,
        )

        print(f"Epoch {epoch}/{args.epochs}:")
        print(
            f"  Train - Loss: {train_stats['loss']:.4f} "
            f"(Attr:{train_stats['loss_attr']:.4f}, Cls:{train_stats['loss_cls']:.4f}, "
            f"Fus:{train_stats['loss_fus']:.4f}, "
            f"Reg:{train_stats['loss_reg']:.4f}, AttrPred:{train_stats['loss_attr_pred']:.4f}, "
            f"Cont:{train_stats.get('loss_continuity', 0.0):.4f}, KernelReg:{train_stats.get('loss_kernel_reg', 0.0):.4f}, ThreshReg:{train_stats.get('loss_threshold_reg', 0.0):.4f}), "
            f"AttrAcc: {train_stats['acc_attr']*100:.2f}%, ClsAcc: {train_stats['acc_cls']*100:.2f}%, "
            f"FusAcc: {train_stats['acc_fus']*100:.2f}%"
        )
        print(
            f"  Val   - Loss: {val_stats['loss']:.4f} "
            f"(Attr:{val_stats['loss_attr']:.4f}, Cls:{val_stats['loss_cls']:.4f}, "
            f"Fus:{val_stats['loss_fus']:.4f}, "
            f"Reg:{val_stats['loss_reg']:.4f}, AttrPred:{val_stats['loss_attr_pred']:.4f}, "
            f"Cont:{val_stats.get('loss_continuity', 0.0):.4f}, KernelReg:{val_stats.get('loss_kernel_reg', 0.0):.4f}, ThreshReg:{val_stats.get('loss_threshold_reg', 0.0):.4f}), "
            f"AttrAcc: {val_stats['acc_attr']*100:.2f}%, ClsAcc: {val_stats['acc_cls']*100:.2f}%, "
            f"FusAcc: {val_stats['acc_fus']*100:.2f}%\n"
        )
        
        # Log metrics to wandb
        if args.use_wandb:
            wandb.log({
            "epoch": epoch,
            "train/loss": train_stats['loss'],
            "train/loss_attr": train_stats['loss_attr'],
            "train/loss_cls": train_stats['loss_cls'],
            "train/loss_fus": train_stats['loss_fus'],
            "train/loss_reg": train_stats['loss_reg'],
            "train/loss_attr_pred": train_stats['loss_attr_pred'],
            "train/loss_continuity": train_stats.get('loss_continuity', 0.0),
            "train/loss_kernel_reg": train_stats.get('loss_kernel_reg', 0.0),
            "train/loss_threshold_reg": train_stats.get('loss_threshold_reg', 0.0),
            "train/acc_attr": train_stats['acc_attr'],
            "train/acc_cls": train_stats['acc_cls'],
            "train/acc_fus": train_stats['acc_fus'],
            "val/loss": val_stats['loss'],
            "val/loss_attr": val_stats['loss_attr'],
            "val/loss_cls": val_stats['loss_cls'],
            "val/loss_fus": val_stats['loss_fus'],
            "val/loss_reg": val_stats['loss_reg'],
            "val/loss_attr_pred": val_stats['loss_attr_pred'],
            "val/loss_continuity": val_stats.get('loss_continuity', 0.0),
            "val/loss_kernel_reg": val_stats.get('loss_kernel_reg', 0.0),
            "val/loss_threshold_reg": val_stats.get('loss_threshold_reg', 0.0),
            "val/acc_attr": val_stats['acc_attr'],
            "val/acc_cls": val_stats['acc_cls'],
            "val/acc_fus": val_stats['acc_fus'],
            "best_val_acc": best_val_acc,
            "learning_rate": scheduler.get_last_lr()[0],
            })

        # Save best model (by validation fused accuracy)
        if val_stats["acc_fus"] > best_val_acc:
            best_val_acc = val_stats["acc_fus"]
            # Save complete checkpoint including class_attr_map to ensure consistency
            checkpoint = {
                'state_dict': model.state_dict(),
                'class_attr_map': class_attr_map.cpu(),  # Save mapping matrix
                'best_val_acc': best_val_acc,
                'epoch': epoch,
            }
            torch.save(checkpoint, best_model_path)
            print(f"  Saved best model to: {best_model_path} (Val FusAcc: {best_val_acc*100:.2f}%)\n")

        # Learning rate scheduling
        scheduler.step()

    print("\nTraining completed! Best validation accuracy: {:.2f}%".format(best_val_acc * 100))

    # 6) Final testing
    print("\n=============== Final Testing ===============")
    if os.path.isfile(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        # Handle both old format (state_dict only) and new format (checkpoint dict)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            # Use saved class_attr_map if available, otherwise use current one
            if 'class_attr_map' in checkpoint:
                saved_map = checkpoint['class_attr_map'].to(device)
                with torch.no_grad():
                    model.class_attr_map.copy_(saved_map)
                print(f"Loaded best model: {best_model_path} (Epoch {checkpoint.get('epoch', 'unknown')}, Val Acc: {checkpoint.get('best_val_acc', 0)*100:.2f}%)\n")
            else:
                with torch.no_grad():
                    model.class_attr_map.copy_(class_attr_map)
                print(f"Loaded best model: {best_model_path}\n")
        else:
            # Old format: just state_dict
            model.load_state_dict(checkpoint)
            with torch.no_grad():
                model.class_attr_map.copy_(class_attr_map)
            print(f"Loaded best model: {best_model_path}\n")
    else:
        print("Warning: best model not found, using last epoch weights.\n")

    test_stats = test(
        model,
        test_loader,
        device,
        attr_embeddings,
        class_attr_map,
        num_classes=num_classes,
    )

    # Force flush to ensure output is written immediately
    import sys
    sys.stdout.flush()
    
    print("=============== Test Results ===============")
    print(f"Overall accuracy (all dimensions):")
    print(f"  AttrAcc: {test_stats['acc_attr']*100:.2f}%")
    print(f"  ClsAcc:  {test_stats['acc_cls']*100:.2f}%")
    print(f"  FusAcc:  {test_stats['acc_fus']*100:.2f}%\n")
    sys.stdout.flush()
    
    # Log test results to wandb
    if args.use_wandb and wandb.run:
        try:
            wandb.log({
                "test/acc_attr": test_stats['acc_attr'],
                "test/acc_cls": test_stats['acc_cls'],
                "test/acc_fus": test_stats['acc_fus'],
            })
            
            # Log per-class accuracies
            for idx in range(num_classes):
                class_name = class_keys[idx] if idx < len(class_keys) else str(idx)
                wandb.log({
                    f"test/per_class_attr/{class_name}": test_stats["per_class_attr"].get(idx, 0.0),
                    f"test/per_class_cls/{class_name}": test_stats["per_class_cls"].get(idx, 0.0),
                    f"test/per_class_fus/{class_name}": test_stats["per_class_fus"].get(idx, 0.0),
                })
        except Exception as e:
            print(f"[WARNING] Failed to log test results to wandb: {e}")

    def print_per_class(title: str, per_class: Dict[int, float]):
        print(f"Per-class accuracy ({title}):")
        for idx in range(num_classes):
            acc = per_class.get(idx, 0.0) * 100
            name = class_keys[idx] if idx < len(class_keys) else str(idx)
            print(f"  {name}: {acc:.2f}%")
        print("")
        sys.stdout.flush()

    print_per_class("Attr-based classification", test_stats["per_class_attr"])
    print_per_class("Direct classification", test_stats["per_class_cls"])
    print_per_class("Fused classification", test_stats["per_class_fus"])

    print("Classification Report (Fused predictions):")
    print(test_stats["report"])
    print("\nConfusion Matrix (Fused predictions):")
    print("Rows=true class, Columns=predicted class")
    print(f"Class order: {class_keys}")
    print(test_stats["confusion_matrix"])
    sys.stdout.flush()

    print("\n==========================================")
    print("Training completed!")
    print(f"End time: {datetime.now().strftime('%a %b %d %I:%M:%S %p %Z %Y')}")
    print("==========================================")
    
    # Finish wandb run and save output files
    if args.use_wandb and wandb.run:
        try:
            # Save only the current run's SLURM output files
            job_id = os.environ.get('SLURM_JOB_ID')
            if job_id:
                print(f"\n[INFO] Attempting to save output files for SLURM job ID: {job_id}")
                sys.stdout.flush()
                
                # Only save output files for current job
                output_files = [f"breast_attr_{job_id}.out", f"breast_attr_{job_id}.err"]
                run_dir = wandb.run.dir if hasattr(wandb.run, 'dir') else None
                if run_dir:
                    # Determine the correct files directory path (save to files/, not code/)
                    if run_dir.endswith('files'):
                        output_save_dir = run_dir  # Already in files directory
                    else:
                        output_save_dir = os.path.join(run_dir, "files")
                    os.makedirs(output_save_dir, exist_ok=True)
                    print(f"[INFO] Saving output files to: {output_save_dir}")
                    sys.stdout.flush()
                    
                    saved_count = 0
                    for output_file in output_files:
                        try:
                            if os.path.exists(output_file):
                                dest_path = os.path.join(output_save_dir, output_file)
                                shutil.copy2(output_file, dest_path)
                                print(f"[INFO] 鉁?Saved output file: {output_file} -> {dest_path}")
                                saved_count += 1
                            else:
                                # Try to find the file with pattern matching (in case job ID format is different)
                                import glob
                                pattern = output_file.replace(job_id, "*")
                                matches = glob.glob(pattern)
                                if matches:
                                    # Use the most recent file
                                    matches.sort(key=os.path.getmtime, reverse=True)
                                    if matches:
                                        dest_path = os.path.join(output_save_dir, os.path.basename(matches[0]))
                                        shutil.copy2(matches[0], dest_path)
                                        print(f"[INFO] 鉁?Saved output file: {os.path.basename(matches[0])} -> {dest_path}")
                                        saved_count += 1
                                else:
                                    print(f"[WARNING] Output file not found: {output_file}")
                                    print(f"  Checked path: {os.path.abspath(output_file)}")
                                    print(f"  Current working directory: {os.getcwd()}")
                        except Exception as e:
                            print(f"[ERROR] Failed to save output file {output_file}: {str(e)}")
                            import traceback
                            traceback.print_exc()
                    
                    if saved_count > 0:
                        print(f"[INFO] Successfully saved {saved_count} output file(s) to wandb run directory")
                    else:
                        print(f"[WARNING] No output files were saved (files may not exist or job ID mismatch)")
                    sys.stdout.flush()
                else:
                    print(f"[WARNING] wandb run directory not found, cannot save output files")
            else:
                print("[WARNING] SLURM_JOB_ID not found in environment variables")
                print("[INFO] Output files will not be automatically saved to wandb")
                sys.stdout.flush()
        except Exception as e:
            print(f"[ERROR] Error while saving output files: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Finish wandb run
        try:
            wandb.finish()
            print("[INFO] Wandb run finished.")
            sys.stdout.flush()
        except Exception as e:
            print(f"[WARNING] Error finishing wandb run: {str(e)}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

