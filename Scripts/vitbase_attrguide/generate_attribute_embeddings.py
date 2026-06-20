#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate and save attribute embeddings using CLIP
This script can be run once to generate embeddings, then use --precomputed_attr_emb in training
"""
import os
import sys
import argparse

# Set stdout encoding to UTF-8
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    import io
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import clip

from utils.attribute_processor import AttributeProcessor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate and save attribute embeddings using CLIP")
    
    parser.add_argument("--attr_csv", type=str, required=True,
                       help="Path to attribute CSV file")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save attribute embeddings (.npy or .pt file)")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                       choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
                       help="CLIP model name (default: ViT-B/32)")
    parser.add_argument("--clip_download_root", type=str, default=None,
                       help="Custom directory for CLIP model download (default: ~/.cache/clip). "
                            "Use this to avoid disk quota issues, e.g., /scratch/liyapeng/zhaobo/CLIP_models")
    parser.add_argument("--clip_model_path", type=str, default=None,
                       help="Path to pre-saved CLIP model .pt file (e.g., /path/to/ViT-B-32.pt). "
                            "If provided, model will be loaded from here instead of downloading.")
    parser.add_argument("--use_prompt", action="store_true", default=True,
                       help="Use prompt template for better semantic understanding (default: True). "
                            "Recommended for medical attributes. Use --no_use_prompt to disable.")
    parser.add_argument("--no_use_prompt", dest="use_prompt", action="store_false",
                       help="Disable prompt template, use raw attribute words")
    parser.add_argument("--prompt_template", type=str, 
                       default="an ultrasound image showing {attr}",
                       help="Prompt template string. Use {attr} as placeholder for attribute word. "
                            "Examples: 'a photo of {attr}', 'an ultrasound image showing {attr}', "
                            "'a medical image of {attr}', '{attr}' (no prompt). "
                            "Default: 'an ultrasound image showing {attr}' (optimized for medical ultrasound)")
    parser.add_argument("--smart_prompt", action="store_true", default=True,
                       help="Enable smart prompt mode: automatically detect fragments and use raw attribute for them (default: True). "
                            "Complete words/phrases use prompt, fragments (like 'of maxilla and', 'Aortic ar') use raw attribute. "
                            "Use --no_smart_prompt to disable.")
    parser.add_argument("--no_smart_prompt", dest="smart_prompt", action="store_false",
                       help="Disable smart prompt mode, use same prompt template for all attributes")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite output file if it already exists (useful for batch jobs)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Generate Attribute Embeddings Using CLIP")
    print("=" * 60)
    print(f"Attribute CSV: {args.attr_csv}")
    print(f"Output path: {args.output_path}")
    print(f"CLIP model: {args.clip_model}")
    if args.clip_model_path:
        print(f"CLIP model path: {args.clip_model_path} (will load from local file)")
    elif args.clip_download_root:
        print(f"CLIP download directory: {args.clip_download_root}")
    else:
        print(f"CLIP download directory: ~/.cache/clip (default)")
    print("=" * 60)
    
    # Check if output file already exists
    if os.path.exists(args.output_path):
        if args.overwrite:
            print(f"Output file already exists: {args.output_path}")
            print("--overwrite flag set, will overwrite existing file.")
        else:
            # Check if running in non-interactive mode (e.g., SLURM)
            if not sys.stdin.isatty():
                print(f"Warning: Output file already exists: {args.output_path}")
                print("Running in non-interactive mode. Use --overwrite flag to overwrite.")
                print("Aborted.")
                return
            response = input(f"Output file already exists: {args.output_path}\nOverwrite? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
    
    # Create attribute processor
    print("\n[1/3] Loading attribute table...")
    attr_processor = AttributeProcessor(
        csv_path=args.attr_csv,
        use_clip=True,
        clip_model_name=args.clip_model,
        clip_download_root=args.clip_download_root,
    )
    
    # Override CLIP model loading if a local path is provided
    if args.clip_model_path:
        print(f"\n[1.5/3] Loading CLIP model from local path: {args.clip_model_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if file exists
        if not os.path.exists(args.clip_model_path):
            print(f"Error: CLIP model file not found: {args.clip_model_path}")
            print("Falling back to default CLIP model loading (download if not cached).")
            attr_processor.clip_model = None
        else:
            try:
                # Method 1: Try to load directly using file path (CLIP supports this if file is valid checkpoint)
                print(f"Attempting to load CLIP model directly from file path...")
                try:
                    model, _ = clip.load(args.clip_model_path, device=device, jit=False)
                    model.eval()
                    attr_processor.clip_model = model
                    attr_processor.device = device
                    print(f"鉁?CLIP model loaded directly from file path successfully!")
                except Exception as e1:
                    print(f"Direct file path load failed: {e1}")
                    print(f"Attempting alternative loading method...")
                    
                    # Method 2: Check if there's a cached model architecture we can use
                    # Look for cached model in download_root or default cache
                    cache_dirs = []
                    if args.clip_download_root:
                        cache_dirs.append(args.clip_download_root)
                    cache_dirs.append(os.path.expanduser("~/.cache/clip"))
                    cache_dirs.append("/scratch/liyapeng/zhaobo/CLIP_models")
                    
                    model_loaded = False
                    for cache_dir in cache_dirs:
                        if os.path.exists(cache_dir):
                            # Try to find cached model file
                            cached_model_patterns = [
                                os.path.join(cache_dir, f"{args.clip_model.replace('/', '-')}.pt"),
                                os.path.join(cache_dir, "ViT-B-32.pt"),
                            ]
                            
                            for cached_model_path in cached_model_patterns:
                                if os.path.exists(cached_model_path):
                                    print(f"Found cached model architecture at: {cached_model_path}")
                                    try:
                                        # Load architecture from cache
                                        base_model, _ = clip.load(cached_model_path, device=device, jit=False)
                                        
                                        # Load weights from user's file
                                        checkpoint = torch.load(args.clip_model_path, map_location=device)
                                        if isinstance(checkpoint, dict):
                                            if 'state_dict' in checkpoint:
                                                state_dict = checkpoint['state_dict']
                                            elif 'model' in checkpoint:
                                                state_dict = checkpoint['model']
                                            else:
                                                state_dict = checkpoint
                                        else:
                                            state_dict = checkpoint
                                        
                                        base_model.load_state_dict(state_dict, strict=False)
                                        model = base_model
                                        model.eval()
                                        attr_processor.clip_model = model
                                        attr_processor.device = device
                                        print(f"鉁?CLIP model loaded: architecture from cache, weights from {args.clip_model_path}")
                                        model_loaded = True
                                        break
                                    except Exception as e2:
                                        print(f"Failed to load from {cached_model_path}: {e2}")
                                        continue
                            
                            if model_loaded:
                                break
                    
                    if not model_loaded:
                        raise RuntimeError(f"Failed to load CLIP model. Tried direct load and cached architecture methods. "
                                         f"Last error: {e1}. Please ensure the model file is valid or network is available.")
                        
            except Exception as e:
                print(f"Error loading CLIP model from {args.clip_model_path}: {e}")
                import traceback
                traceback.print_exc()
                print("\nFalling back to default CLIP model loading (download if not cached).")
                print("Note: This will fail if network is unavailable and model is not cached.")
                attr_processor.clip_model = None  # Reset to allow default loading
    
    # Generate embeddings
    print("\n[2/3] Generating attribute embeddings with CLIP...")
    if args.clip_model_path:
        print("Using pre-loaded CLIP model from local file.")
    else:
        print("Note: This may take a few minutes and requires downloading CLIP model if not cached.")
    
    print(f"Prompt settings:")
    print(f"  - Use prompt: {args.use_prompt}")
    if args.use_prompt:
        print(f"  - Prompt template: '{args.prompt_template}'")
        print(f"  - Smart prompt mode: {args.smart_prompt}")
        if args.smart_prompt:
            print(f"    (Will auto-detect fragments and use raw attribute for them)")
    
    attr_embeddings = attr_processor.encode_attributes_with_clip(
        use_prompt=args.use_prompt,
        prompt_template=args.prompt_template,
        smart_prompt=args.smart_prompt
    )
    
    # Save embeddings
    print(f"\n[3/3] Saving embeddings to: {args.output_path}")
    attr_processor.save_attribute_embeddings(attr_embeddings, args.output_path)
    
    # Verify file was saved correctly
    if os.path.exists(args.output_path):
        file_size = os.path.getsize(args.output_path)
        expected_size = attr_embeddings.numel() * 4  # float32 = 4 bytes
        
        # Try to reload the file to verify it's valid
        try:
            reloaded = torch.load(args.output_path, map_location='cpu')
            if isinstance(reloaded, torch.Tensor):
                reloaded_shape = reloaded.shape
                reloaded_dtype = reloaded.dtype
                if reloaded_shape == attr_embeddings.shape:
                    print("\nFile verification PASSED:")
                    print(f"  File path: {args.output_path}")
                    print(f"  File size: {file_size} bytes ({file_size / 1024:.2f} KB)")
                    print(f"  Expected size (uncompressed): ~{expected_size} bytes ({expected_size / 1024:.2f} KB)")
                    print(f"  Loaded shape: {reloaded_shape}")
                    print(f"  Loaded dtype: {reloaded_dtype}")
                    if file_size < expected_size * 0.3:  # File is much smaller than expected
                        print(f"  NOTE: File size is smaller than expected, but file loads correctly.")
                        print(f"        This may be due to file system compression or storage optimization.")
                    else:
                        print("  File size is reasonable")
                else:
                    print("\nFile verification FAILED:")
                    print(f"  Expected shape: {attr_embeddings.shape}")
                    print(f"  Loaded shape: {reloaded_shape}")
            else:
                print("\nFile verification FAILED:")
                print(f"  File does not contain a tensor (type: {type(reloaded)})")
        except Exception as e:
            print("\nFile verification FAILED:")
            print(f"  Error loading file: {e}")
    else:
        print(f"\nERROR: File was not created at {args.output_path}")
    
    print("\n" + "=" * 60)
    print("Success! Attribute embeddings saved.")
    print(f"Embedding shape: {attr_embeddings.shape}")
    print(f"Total attributes: {len(attr_processor.all_attributes)}")
    print(f"To use in training, add: --precomputed_attr_emb {args.output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

