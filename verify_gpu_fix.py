#!/usr/bin/env python3
"""Verify GPU fix for Qwen3-VL ONNX export."""

import torch
from convert.converters import QwenConverter

print("=" * 60)
print("Qwen3-VL GPU Fix Verification")
print("=" * 60)

# 1. Check CUDA availability
print("\n1. CUDA Availability:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   Total VRAM: {total_mem:.1f}GB")

# 2. Test auto-detect (should use CUDA on L4 systems)
print("\n2. Auto-detect Test:")
converter_auto = QwenConverter(use_fp16=True, device=None)
print(f"   Device selected: {converter_auto.device}")
print(f"   Expected: 'cuda' (on L4 GPU systems)")

# 3. Test explicit CUDA
if torch.cuda.is_available():
    print("\n3. Explicit CUDA Test:")
    converter_cuda = QwenConverter(use_fp16=True, device="cuda")
    print(f"   Device: {converter_cuda.device}")

    # Test GPU memory check
    print("\n4. GPU Memory Check:")
    converter_cuda._check_gpu_memory()

# 4. Test explicit CPU
print("\n5. Explicit CPU Test:")
converter_cpu = QwenConverter(use_fp16=True, device="cpu")
print(f"   Device: {converter_cpu.device}")

print("\n" + "=" * 60)
print("✅ Verification complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Run actual ONNX export with: --device cuda")
print("2. Should see: 'Using device_map=\"auto\" for direct GPU loading'")
print("3. Should complete without OOM at 47%")
print("=" * 60)
