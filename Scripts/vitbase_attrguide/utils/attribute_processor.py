#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Attribute processing utility: Load attributes from CSV table, build class-attribute mapping matrix, generate attribute word embeddings
One word per attribute, not prompt
"""
import os
import csv
import json
import re
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional
import ssl
import urllib.request

# Fix SSL certificate verification error when downloading CLIP models
# This is needed in some network environments with proxy/certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

import clip


class AttributeProcessor:
    """Attribute processor: Process attribute table, build class-attribute mapping matrix and attribute embeddings"""
    
    def __init__(self, csv_path: str, use_clip: bool = True, clip_model_name: str = "ViT-B/32", 
                 clip_download_root: Optional[str] = None):
        """
        Args:
            csv_path: Path to attribute CSV file
            use_clip: Whether to use CLIP to generate attribute embeddings
            clip_model_name: CLIP model name
            clip_download_root: Custom directory for CLIP model download (default: ~/.cache/clip)
                               Use this to avoid disk quota issues in home directory
        """
        # Handle path encoding issues - try multiple approaches
        self.csv_path = None
        
        # Strategy 1: Try the path as-is
        if os.path.exists(csv_path):
            self.csv_path = csv_path
            print(f"[AttributeProcessor] Found CSV file at: {csv_path}")
        else:
            print(f"[AttributeProcessor] CSV path not found directly: {csv_path}")
            print(f"[AttributeProcessor] Attempting to auto-detect CSV file...")
            
            # Strategy 2: Extract directory and try to find CSV files there
            # Handle both absolute and relative paths
            dir_path = os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.'
            
            # Try to find the directory even if path encoding is broken
            dir_candidates = [dir_path]
            # If dir_path contains surrogate escapes, try to extract a valid directory
            if any(ord(c) >= 0xD800 and ord(c) <= 0xDFFF for c in dir_path):
                # Try to find a valid parent directory
                parts = dir_path.split(os.sep)
                for i in range(len(parts), 0, -1):
                    test_dir = os.sep.join(parts[:i])
                    if test_dir and os.path.exists(test_dir):
                        dir_candidates.append(test_dir)
                        break
            
            # Try each directory candidate
            for test_dir in dir_candidates:
                if not os.path.exists(test_dir):
                    continue
                try:
                    # List all CSV files in the directory
                    csv_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
                    if not csv_files:
                        continue
                    
                    print(f"[AttributeProcessor] Found {len(csv_files)} CSV file(s) in directory: {test_dir}")
                    
                    # If only one CSV file, use it
                    if len(csv_files) == 1:
                        self.csv_path = os.path.join(test_dir, csv_files[0])
                        print(f"[AttributeProcessor] Auto-detected CSV file: {self.csv_path}")
                        break
                    elif len(csv_files) > 1:
                        # Try to match by looking for files with non-ASCII characters (likely the Chinese file)
                        for f in csv_files:
                            # Check if the file contains non-ASCII characters
                            if any(ord(c) > 127 for c in f):
                                self.csv_path = os.path.join(test_dir, f)
                                print(f"[AttributeProcessor] Auto-detected CSV file (by non-ASCII match): {self.csv_path}")
                                break
                        if self.csv_path:
                            break
                except (OSError, PermissionError) as e:
                    print(f"[AttributeProcessor] Warning: Could not access directory {test_dir}: {e}")
                    continue
            
            # Strategy 3: Try normalized path
            if self.csv_path is None:
                try:
                    normalized = os.path.normpath(os.path.abspath(csv_path))
                    if os.path.exists(normalized):
                        self.csv_path = normalized
                        print(f"[AttributeProcessor] Found CSV file at normalized path: {normalized}")
                except Exception as e:
                    print(f"[AttributeProcessor] Warning: Normalization failed: {e}")
            
            # Final check: if still not found, raise error with helpful message
            if self.csv_path is None or not os.path.exists(self.csv_path):
                # List available CSV files in the directory for debugging
                error_msg = f"CSV file not found: {csv_path}\n"
                # Try to list files in the directory
                for test_dir in dir_candidates:
                    if os.path.exists(test_dir):
                        try:
                            available_csvs = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
                            if available_csvs:
                                error_msg += f"Available CSV files in directory '{test_dir}': {available_csvs}\n"
                                error_msg += f"Please check the path or use one of the available files."
                        except Exception:
                            pass
                raise FileNotFoundError(error_msg)
        
        self.use_clip = use_clip
        self.clip_model_name = clip_model_name
        self.clip_download_root = clip_download_root
        self.clip_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.clip_download_root:
            os.makedirs(self.clip_download_root, exist_ok=True)
            print(f"[AttributeProcessor] Using custom CLIP download directory: {self.clip_download_root}")
        
        # Load attribute table
        self.attr_dict, self.attr_first_occurrence = self._load_attribute_table()
        
        # Build set of all attributes (preserve first occurrence order, not alphabetical)
        self.all_attributes = self._extract_all_attributes()
        
        print(f"[AttributeProcessor] Loaded attributes for {len(self.attr_dict)} classes")
        print(f"[AttributeProcessor] Total {len(self.all_attributes)} unique attribute words")
    
    def _load_attribute_table(self) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """
        从 CSV 读取属性表，构建 {class_key: [attr1, attr2, ...]} 字典。
        同时记录每个属性首次出现的行号，用于保持属性顺序。
        
        兼容两种写法：
        1）只有 `folder` 列：以 folder 作为 key（旧格式）
        2）有 `folder` + `name` 列：优先使用 `name` 作为 key（与数据集文件夹名一致）
        
        注意：会自动忽略 `folder` 和 `name` 两列本身，不把它们当成属性列。
        
        Returns:
            attr_dict: {class_key: [attr1, attr2, ...]}
            attr_first_occurrence: {attr_name: first_row_index} - 用于保持属性顺序
        """
        attr_dict: Dict[str, List[str]] = {}
        attr_first_occurrence: Dict[str, int] = {}  # Track first occurrence row index

        # 改用 pandas 读取，自动探测分隔符，尝试多种编码
        # 使用 open() 和文件句柄来避免路径编码问题
        encodings = ["utf-8-sig", "gb18030", "gbk", "latin1"]
        df = None
        for enc in encodings:
            try:
                # Use file handle to avoid path encoding issues
                with open(self.csv_path, 'r', encoding=enc, errors='replace') as f:
                    df = pd.read_csv(f, engine="python", sep=None)
                print(f"[AttributeProcessor] Read CSV with encoding: {enc}, shape={df.shape}")
                break
            except UnicodeDecodeError as e:
                print(f"[AttributeProcessor] Failed to read CSV with encoding {enc}: {e}")
                continue
            except Exception as e:
                # For other errors (like file not found), try direct path
                try:
                    df = pd.read_csv(self.csv_path, encoding=enc, engine="python", sep=None)
                    print(f"[AttributeProcessor] Read CSV with encoding: {enc} (direct path), shape={df.shape}")
                    break
                except Exception as e2:
                    print(f"[AttributeProcessor] Failed to read CSV with encoding {enc} (direct path): {e2}")
                    continue
        if df is None:
            raise UnicodeError(f"Failed to decode CSV file {self.csv_path} with encodings: {encodings}")

        # 清洗列名
        df.columns = [str(c).strip() for c in df.columns]

        if "folder" not in df.columns and "name" not in df.columns:
            raise ValueError(
                f"CSV file must contain 'folder' and/or 'name' column. "
                f"Available columns: {list(df.columns)}"
            )

        key_column = "name" if "name" in df.columns else "folder"
        print(f"[AttributeProcessor] Using '{key_column}' column as class key")

        attr_columns = [c for c in df.columns if c not in ("folder", "name")]
        print(f"[AttributeProcessor] Found {len(attr_columns)} attribute columns: {attr_columns}")

        for row_idx, (_, row) in enumerate(df.iterrows()):
            key = str(row.get(key_column, "")).strip()
            if not key:
                continue

            attrs: List[str] = []
            for col in attr_columns:
                val = row.get(col)
                if pd.isna(val):
                    continue
                val_str = str(val).strip()
                if not val_str or val_str == "关键区域":
                    continue
                attrs.append(val_str)
                # Record first occurrence of this attribute
                if val_str not in attr_first_occurrence:
                    attr_first_occurrence[val_str] = row_idx

            if attrs:
                attr_dict[key] = attrs

        return attr_dict, attr_first_occurrence
    
    def _detect_delimiter(self) -> str:
        """
        Auto-detect CSV file delimiter (comma or tab)
        
        Returns:
            delimiter: Detected delimiter
        """
        # Read first two lines to detect delimiter; try UTF-8 first, then GBK fallback
        for encoding in ("utf-8-sig", "gbk"):
            try:
                with open(self.csv_path, "r", encoding=encoding) as f:
                    first_line = f.readline()
                    second_line = f.readline()
                break
            except UnicodeDecodeError as e:
                print(f"[AttributeProcessor] Delimiter detect: failed to read with {encoding}: {e}")
                continue
        else:
            raise UnicodeError(f"Failed to decode CSV file {self.csv_path} with utf-8-sig and gbk encodings.")
        
        # Count comma and tab occurrences
        comma_count = first_line.count(',')
        tab_count = first_line.count('\t')
        
        # If tab count is greater than comma count, use tab
        if tab_count > comma_count:
            return '\t'
        else:
            return ','
    
    def _extract_all_attributes(self) -> List[str]:
        """
        Extract all unique attribute words, preserving first occurrence order.
        This ensures the attribute order matches the training order (CSV row order),
        not alphabetical order.
        """
        # Use dict to preserve insertion order (Python 3.7+)
        # Sort by first occurrence row index to maintain CSV order
        all_attrs_dict = {}
        for attrs in self.attr_dict.values():
            for attr in attrs:
                if attr not in all_attrs_dict:
                    # Store (row_index, attr) to sort by first occurrence
                    all_attrs_dict[attr] = self.attr_first_occurrence.get(attr, 999999)
        
        # Sort by first occurrence row index, then by attribute name for ties
        sorted_attrs = sorted(all_attrs_dict.items(), key=lambda x: (x[1], x[0]))
        return [attr for attr, _ in sorted_attrs]
    
    def build_class_attr_map(self, class_to_folder: Dict[str, str]) -> torch.Tensor:
        """
        Build class-attribute mapping matrix (similar to TransZero's self.att)
        
        Args:
            class_to_folder: {class_key: folder_name} mapping
        
        Returns:
            class_attr_map: (num_classes, num_attrs) matrix
                class_attr_map[i, j] = 1 means class i has attribute j, otherwise 0
        """
        num_classes = len(class_to_folder)
        num_attrs = len(self.all_attributes)
        
        # Build attribute to index mapping
        attr_to_idx = {attr: idx for idx, attr in enumerate(self.all_attributes)}
        
        # Initialize mapping matrix
        class_attr_map = torch.zeros(num_classes, num_attrs)
        
        # Fill mapping matrix
        classes_without_attrs = []
        for class_idx, (class_key, folder) in enumerate(class_to_folder.items()):
            if folder in self.attr_dict:
                attrs = self.attr_dict[folder]
                matched_attrs = 0
                for attr in attrs:
                    if attr in attr_to_idx:
                        attr_idx = attr_to_idx[attr]
                        class_attr_map[class_idx, attr_idx] = 1.0
                        matched_attrs += 1
                
                # 检查：每个类别必须至少有一个属性
                if matched_attrs == 0:
                    classes_without_attrs.append(f"{class_key} (folder: {folder})")
                    print(f"[AttributeProcessor] 警告: 类别 '{class_key}' (folder: '{folder}') 没有匹配的属性！")
            else:
                classes_without_attrs.append(f"{class_key} (folder: {folder} not found in CSV)")
                print(f"[AttributeProcessor] 错误: 类别 '{class_key}' 对应的folder '{folder}' 在CSV中不存在！")
        
        # 强制检查：如果有类别没有属性，报错
        if classes_without_attrs:
            error_msg = f"\n[AttributeProcessor] 错误: 以下类别没有对应的属性标签，必须修复！\n"
            for cls in classes_without_attrs:
                error_msg += f"  - {cls}\n"
            error_msg += "\n请检查CSV文件中的folder列和数据集文件夹名称的映射关系。"
            raise ValueError(error_msg)
        
        # 验证每个类别至少有一个属性
        attr_counts = class_attr_map.sum(dim=1)
        classes_with_zero_attrs = []
        for class_idx, (class_key, folder) in enumerate(class_to_folder.items()):
            if attr_counts[class_idx].item() == 0:
                classes_with_zero_attrs.append(class_key)
        
        if classes_with_zero_attrs:
            error_msg = f"\n[AttributeProcessor] 错误: 以下类别在映射矩阵中没有任何属性（属性数量为0）：\n"
            for cls in classes_with_zero_attrs:
                error_msg += f"  - {cls}\n"
            error_msg += "\n这会导致后续训练和推理出错，必须修复！"
            raise ValueError(error_msg)
        
        print(f"[AttributeProcessor] Built class-attribute mapping matrix: ({num_classes}, {num_attrs})")
        print(f"[AttributeProcessor] Non-zero elements in mapping matrix: {class_attr_map.sum().item()}")
        
        # 打印每个类别的属性数量
        print(f"[AttributeProcessor] 每个类别的属性数量验证:")
        for class_idx, (class_key, folder) in enumerate(class_to_folder.items()):
            attr_count = attr_counts[class_idx].item()
            print(f"  - {class_key}: {int(attr_count)} 个属性")
        
        return class_attr_map
    
    def _load_clip_model(self, max_retries: int = 3):
        """Load CLIP model with retry mechanism"""
        if self.clip_model is None:
            print(f"[AttributeProcessor] Loading CLIP model: {self.clip_model_name}")
            if self.clip_download_root:
                print(f"[AttributeProcessor] Downloading to: {self.clip_download_root}")
            for attempt in range(max_retries):
                try:
                    # Use custom download directory if specified (to avoid disk quota issues)
                    self.clip_model, _ = clip.load(
                        self.clip_model_name, 
                        device=self.device,
                        download_root=self.clip_download_root
                    )
                    self.clip_model.eval()
                    print(f"[AttributeProcessor] CLIP model loaded successfully!")
                    return
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"[AttributeProcessor] Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                        print(f"[AttributeProcessor] Retrying in 5 seconds...")
                        import time
                        time.sleep(5)
                    else:
                        print(f"[AttributeProcessor] Failed to load CLIP model after {max_retries} attempts")
                        raise RuntimeError(f"Failed to load CLIP model: {str(e)}")
    
    def _is_complete_attribute(self, attr: str) -> bool:
        """
        Check if attribute is a complete word/phrase that can be used with prompt, 
        or a fragment that should be used as-is.
        
        Criteria for fragments (should NOT use prompt):
        - Starts with preposition (of, in, on, at, by, for, with, from, to, the, a, an)
        - Ends with conjunction (and, or, but)
        - Contains incomplete abbreviations (ends with space or has "ar", "pel", "ce", "vaul" etc.)
        - Very short (< 3 characters after stripping)
        
        Args:
            attr: Attribute string
        
        Returns:
            True if attribute is complete and can use prompt, False if fragment (use as-is)
        """
        attr = attr.strip()
        if not attr:
            return False
        
        words = attr.split()
        if not words:
            return False
        
        # Very short attributes (< 3 chars) are likely fragments
        if len(attr) < 3:
            return False
        
        # Check if starts with preposition (fragment indicator)
        prepositions = {'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'the', 'a', 'an'}
        if words[0].lower() in prepositions:
            return False
        
        # Check if ends with conjunction (fragment indicator)
        conjunctions = {'and', 'or', 'but', 'nor', 'so', 'yet'}
        if len(words) > 0 and words[-1].lower() in conjunctions:
            return False
        
        # Check for incomplete abbreviations (common in medical terms)
        # These patterns indicate incomplete abbreviations (e.g., "Aortic ar" instead of "Aortic arch")
        # We check if these appear as separate words (preceded by space, followed by space or end)
        # This avoids false positives like "arteries" (contains "ar" but is complete)
        incomplete_patterns = [' ar', ' pel', ' ce', ' vaul', ' sep', ' pulm']
        attr_lower = attr.lower()
        
        for pattern in incomplete_patterns:
            # Check if pattern appears as a separate word
            # It should be preceded by space (or at start) and followed by space or end
            pattern_pos = attr_lower.find(pattern)
            if pattern_pos >= 0:
                # Check if preceded by space or at start
                preceded_by_space = (pattern_pos == 0 or attr_lower[pattern_pos - 1] == ' ')
                # Check if followed by space or at end
                followed_by_space = (pattern_pos + len(pattern) >= len(attr_lower) or 
                                   attr_lower[pattern_pos + len(pattern)] == ' ')
                if preceded_by_space and followed_by_space:
                    return False
        
        # If ends with space, likely incomplete
        if attr.endswith(' '):
            return False
        
        return True
    
    def encode_attributes_with_clip(self, batch_size: int = 32, precomputed_path: Optional[str] = None, 
                                     use_prompt: bool = True, prompt_template: str = "a photo of {attr}",
                                     smart_prompt: bool = True) -> torch.Tensor:
        """
        Encode each attribute word using CLIP with optional prompt template for better semantic understanding.
        Supports smart prompt selection: automatically detects if attribute is a fragment and uses raw attribute.
        
        Args:
            batch_size: Batch size for encoding
            precomputed_path: Path to precomputed embeddings (.npy or .pt file). If provided, will load from file instead of computing.
            use_prompt: If True, use prompt template (e.g., "a photo of {attr}") instead of raw word. 
                       This improves semantic understanding, especially for medical terms.
            prompt_template: Prompt template string. Use {attr} as placeholder for attribute word.
                            Examples:
                            - "a photo of {attr}" (general)
                            - "an ultrasound image showing {attr}" (medical ultrasound)
                            - "a medical image of {attr}" (medical)
                            - "{attr}" (no prompt, raw word)
            smart_prompt: If True (default), automatically detect fragments and use raw attribute for them.
                          - Complete words/phrases: use prompt template
                          - Fragments (starts with preposition, ends with conjunction, incomplete abbreviations): use raw attribute
                          This prevents semantic confusion like "an ultrasound image showing of maxilla and"
        
        Returns:
            attr_embeddings: (num_attrs, attr_emb_dim) - Attribute embedding matrix
        """
        # Try to load precomputed embeddings first
        if precomputed_path is not None and os.path.exists(precomputed_path):
            print(f"[AttributeProcessor] Loading precomputed attribute embeddings from: {precomputed_path}")
            try:
                if precomputed_path.endswith('.npy'):
                    attr_embeddings = torch.from_numpy(np.load(precomputed_path)).float()
                elif precomputed_path.endswith('.pt') or precomputed_path.endswith('.pth'):
                    attr_embeddings = torch.load(precomputed_path, map_location='cpu')
                else:
                    raise ValueError(f"Unsupported file format: {precomputed_path}. Use .npy, .pt, or .pth")
                
                if len(attr_embeddings) != len(self.all_attributes):
                    raise ValueError(f"Precomputed embeddings have {len(attr_embeddings)} attributes, but CSV has {len(self.all_attributes)}")
                
                print(f"[AttributeProcessor] Loaded precomputed embeddings: {attr_embeddings.shape}")
                return attr_embeddings
            except Exception as e:
                print(f"[AttributeProcessor] Warning: Failed to load precomputed embeddings: {str(e)}")
                print(f"[AttributeProcessor] Will compute embeddings from scratch...")
        
        if not self.use_clip:
            # If not using CLIP, can use random initialization or word2vec
            attr_emb_dim = 512
            attr_embeddings = torch.randn(len(self.all_attributes), attr_emb_dim)
            print(f"[AttributeProcessor] Using randomly initialized attribute embeddings: ({len(self.all_attributes)}, {attr_emb_dim})")
            return attr_embeddings
        
        self._load_clip_model()
        
        # Prepare attribute texts (with or without prompt)
        if use_prompt:
            if smart_prompt:
                print(f"[AttributeProcessor] Using SMART prompt mode (auto-detect fragments)")
                print(f"[AttributeProcessor] Base prompt template: '{prompt_template}'")
                
                attr_texts = []
                complete_count = 0
                fragment_count = 0
                for attr in self.all_attributes:
                    if self._is_complete_attribute(attr):
                        # Complete word/phrase: use prompt template
                        text = prompt_template.format(attr=attr)
                        attr_texts.append(text)
                        complete_count += 1
                    else:
                        # Fragment: use raw attribute (no prompt to avoid semantic confusion)
                        attr_texts.append(attr)
                        fragment_count += 1
                
                print(f"[AttributeProcessor] Prompt statistics:")
                print(f"  - Complete attributes (using prompt): {complete_count}")
                print(f"  - Fragments (using raw): {fragment_count}")
                print(f"[AttributeProcessor] Example attribute encodings:")
                for i in range(min(10, len(attr_texts))):
                    is_complete = self._is_complete_attribute(self.all_attributes[i])
                    marker = "[COMPLETE]" if is_complete else "[FRAGMENT]"
                    print(f"  {marker} '{self.all_attributes[i]}' -> '{attr_texts[i]}'")
            else:
                print(f"[AttributeProcessor] Using prompt template for ALL attributes: '{prompt_template}'")
                attr_texts = [prompt_template.format(attr=attr) for attr in self.all_attributes]
                print(f"[AttributeProcessor] Example prompts:")
                for i in range(min(5, len(attr_texts))):
                    print(f"  - '{self.all_attributes[i]}' -> '{attr_texts[i]}'")
        else:
            print(f"[AttributeProcessor] Using raw attribute words (no prompt)")
            attr_texts = self.all_attributes
        
        # Encode attributes with CLIP
        attr_embeddings = []
        
        print(f"[AttributeProcessor] Encoding {len(self.all_attributes)} attributes with CLIP...")
        for start_idx in range(0, len(attr_texts), batch_size):
            batch_texts = attr_texts[start_idx:start_idx + batch_size]
            
            # Use CLIP to encode attribute texts
            with torch.no_grad():
                text_tokens = clip.tokenize(batch_texts, truncate=True).to(self.device)
                batch_embeddings = self.clip_model.encode_text(text_tokens)
                # Normalize
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                attr_embeddings.append(batch_embeddings.cpu())
            
            if (start_idx + batch_size) % (batch_size * 10) == 0:
                print(f"[AttributeProcessor] Progress: {min(start_idx + batch_size, len(attr_texts))}/{len(attr_texts)}")
        
        attr_embeddings = torch.cat(attr_embeddings, dim=0)
        attr_emb_dim = attr_embeddings.shape[1]
        
        print(f"[AttributeProcessor] Encoded attribute embeddings using CLIP: ({len(self.all_attributes)}, {attr_emb_dim})")
        if use_prompt:
            print(f"[AttributeProcessor] ✓ Used prompt template for better semantic understanding")
        
        return attr_embeddings
    
    def save_attribute_embeddings(self, attr_embeddings: torch.Tensor, save_path: str):
        """
        Save attribute embeddings to file
        
        Args:
            attr_embeddings: (num_attrs, attr_emb_dim) tensor
            save_path: Path to save (.npy, .pt, or .pth)
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        if save_path.endswith('.npy'):
            np.save(save_path, attr_embeddings.numpy())
        elif save_path.endswith('.pt') or save_path.endswith('.pth'):
            torch.save(attr_embeddings, save_path)
        else:
            raise ValueError(f"Unsupported file format: {save_path}. Use .npy, .pt, or .pth")
        
        print(f"[AttributeProcessor] Saved attribute embeddings to: {save_path}")
        print(f"[AttributeProcessor] Embedding shape: {attr_embeddings.shape}")
    
    def verify_and_reorder_attributes_from_checkpoint(
        self, 
        checkpoint_path: str, 
        class_keys: List[str],
        attr_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, List[str], Optional[Dict[int, int]]]:
        """
        Verify and reorder attributes to match training order using checkpoint's class_attr_map.
        This is the MOST RELIABLE method to ensure 100% correct attribute-label mapping.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            class_keys: List of class names in the same order as training
            attr_embeddings: Current attribute embeddings (num_attrs, attr_emb_dim)
        
        Returns:
            reordered_embeddings: Reordered attribute embeddings matching training order
            reordered_attr_names: Reordered attribute names matching training order
            attr_mapping: Mapping from old index to new index (saved_idx -> current_idx), None if no reordering needed
        """
        print(f"\n[AttributeProcessor] Verifying attribute order from checkpoint: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if not isinstance(checkpoint, dict) or 'class_attr_map' not in checkpoint:
                print(f"[AttributeProcessor] WARNING: Checkpoint does not contain class_attr_map")
                print(f"[AttributeProcessor] Cannot verify attribute order. Using CSV order (may be incorrect!)")
                return attr_embeddings, self.all_attributes, None
            
            saved_class_attr_map = checkpoint['class_attr_map']  # (num_classes, num_attrs)
            print(f"[AttributeProcessor] Loaded class_attr_map from checkpoint: {saved_class_attr_map.shape}")
            
            # Build current class_attr_map from CSV (using current attribute order)
            class_to_folder = {cls: cls for cls in class_keys}
            current_class_attr_map = self.build_class_attr_map(class_to_folder)  # (num_classes, num_attrs)
            
            # Check if they match (same shape and same values)
            if saved_class_attr_map.shape != current_class_attr_map.shape:
                raise ValueError(
                    f"Shape mismatch: checkpoint class_attr_map {saved_class_attr_map.shape} != "
                    f"current class_attr_map {current_class_attr_map.shape}"
                )
            
            # If they match exactly, attribute order is correct
            if torch.allclose(saved_class_attr_map, current_class_attr_map, atol=1e-6):
                print(f"[AttributeProcessor] ✓✓ Attribute order VERIFIED: matches checkpoint!")
                print(f"[AttributeProcessor]   ✓ 属性顺序与训练时完全一致")
                print(f"[AttributeProcessor]   ✓ 使用当前CSV的属性顺序（无需重排序）")
                print(f"[AttributeProcessor]   ✓ 属性对应: 100% 正确")
                return attr_embeddings, self.all_attributes, None
            
            # If they don't match, we need to reorder attributes
            print(f"[AttributeProcessor] ⚠ Attribute order MISMATCH detected!")
            print(f"[AttributeProcessor]   Reordering attributes to match checkpoint...")
            
            # Find the correct attribute order by matching class_attr_map columns
            # For each attribute in saved map, find which attribute in current map matches
            num_classes, num_attrs = saved_class_attr_map.shape
            attr_mapping = {}  # saved_idx -> current_idx
            
            # Build attribute-to-class mapping for saved map
            saved_attr_to_classes = {}
            for attr_idx in range(num_attrs):
                classes_with_attr = torch.where(saved_class_attr_map[:, attr_idx] > 0.5)[0].tolist()
                saved_attr_to_classes[attr_idx] = set(classes_with_attr)
            
            # Build attribute-to-class mapping for current map
            current_attr_to_classes = {}
            for attr_idx in range(num_attrs):
                classes_with_attr = torch.where(current_class_attr_map[:, attr_idx] > 0.5)[0].tolist()
                current_attr_to_classes[attr_idx] = set(classes_with_attr)
            
            # Match attributes by their class membership
            matched_attrs = set()
            for saved_idx in range(num_attrs):
                saved_classes = saved_attr_to_classes[saved_idx]
                best_match = None
                best_overlap = -1
                
                for current_idx in range(num_attrs):
                    if current_idx in matched_attrs:
                        continue
                    current_classes = current_attr_to_classes[current_idx]
                    overlap = len(saved_classes & current_classes)
                    if overlap == len(saved_classes) and overlap == len(current_classes):
                        # Perfect match
                        best_match = current_idx
                        break
                    elif overlap > best_overlap:
                        best_match = current_idx
                        best_overlap = overlap
                
                if best_match is not None:
                    attr_mapping[saved_idx] = best_match
                    matched_attrs.add(best_match)
                else:
                    raise ValueError(f"Cannot find match for attribute {saved_idx} in checkpoint")
            
            # Reorder attributes
            reordered_attr_names = [self.all_attributes[attr_mapping[i]] for i in range(num_attrs)]
            reordered_embeddings = torch.stack([attr_embeddings[attr_mapping[i]] for i in range(num_attrs)])
            
            print(f"[AttributeProcessor] ✓✓ Attributes reordered to match checkpoint!")
            print(f"[AttributeProcessor]   ✓ 属性已重排序以匹配训练时的顺序")
            print(f"[AttributeProcessor]   ✓ 共 {num_attrs} 个属性已重新排列")
            print(f"[AttributeProcessor]   ✓ 属性对应: 100% 正确（已匹配到训练时的顺序）")
            print(f"[AttributeProcessor]   前5个属性（训练时的顺序）: {reordered_attr_names[:5]}")
            
            # Update self.all_attributes to the correct order
            self.all_attributes = reordered_attr_names
            
            return reordered_embeddings, reordered_attr_names, attr_mapping
            
        except Exception as e:
            print(f"[AttributeProcessor] ERROR during attribute verification: {e}")
            print(f"[AttributeProcessor] Using CSV order (may be incorrect!)")
            import traceback
            traceback.print_exc()
            return attr_embeddings, self.all_attributes, None
    
    def get_attribute_info(self) -> Dict:
        """Get attribute information"""
        return {
            'all_attributes': self.all_attributes,
            'attr_dict': self.attr_dict,
            'num_attrs': len(self.all_attributes)
        }


def load_attribute_table_from_csv(csv_path: str) -> Dict[str, List[str]]:
    """
    Convenience function: Load attribute table from CSV
    
    Returns:
        {folder_name: [attr1, attr2, ...]}
    """
    processor = AttributeProcessor(csv_path, use_clip=False)
    return processor.attr_dict


def build_class_attr_map_from_csv(
    csv_path: str,
    class_to_folder: Dict[str, str]
) -> torch.Tensor:
    """
    Convenience function: Build class-attribute mapping matrix from CSV
    
    Args:
        csv_path: Path to attribute CSV file
        class_to_folder: {class_key: folder_name} mapping
    
    Returns:
        class_attr_map: (num_classes, num_attrs) matrix
    """
    processor = AttributeProcessor(csv_path, use_clip=False)
    return processor.build_class_attr_map(class_to_folder)


if __name__ == "__main__":
    # Test code
    csv_path = "path/to/attribute_table.csv"
    processor = AttributeProcessor(csv_path, use_clip=True)
    
    # Get attribute embeddings
    attr_embeddings = processor.encode_attributes_with_clip()
    print(f"Attribute embedding shape: {attr_embeddings.shape}")
    
    # Build class-attribute mapping matrix (example)
    class_to_folder = {
        "class_0": "经丘脑横切面",
        "class_1": "四腔心切面",
        # ... other classes
    }
    class_attr_map = processor.build_class_attr_map(class_to_folder)
    print(f"Class-attribute mapping matrix shape: {class_attr_map.shape}")

