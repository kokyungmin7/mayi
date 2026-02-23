#!/usr/bin/env python3
"""Inspect Qwen3-VL model structure to find vision encoder."""

import torch
from transformers import AutoModelForImageTextToText

print("=" * 80)
print("Qwen3-VL 모델 구조 검사")
print("=" * 80)

# Load model (minimal loading to save memory)
print("\n모델 로딩 중...")
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-8B-Thinking",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("\n✅ 모델 로딩 완료")

# 1. Print model type
print(f"\n모델 타입: {type(model).__name__}")

# 2. Print top-level attributes
print("\n" + "=" * 80)
print("Top-level 속성들:")
print("=" * 80)
for name in dir(model):
    if not name.startswith('_'):
        attr = getattr(model, name, None)
        if isinstance(attr, torch.nn.Module):
            print(f"  ✓ {name}: {type(attr).__name__}")

# 3. Check common vision encoder names
print("\n" + "=" * 80)
print("Vision 인코더 후보 검사:")
print("=" * 80)

candidates = [
    'visual',
    'vision_model',
    'vision_tower',
    'vision_encoder',
    'img_encoder',
    'image_encoder',
    'transformer',
    'model',
]

found_vision = None
for candidate in candidates:
    if hasattr(model, candidate):
        attr = getattr(model, candidate)
        print(f"  ✓ '{candidate}' 발견: {type(attr).__name__}")
        if found_vision is None and isinstance(attr, torch.nn.Module):
            found_vision = candidate
    else:
        print(f"  ✗ '{candidate}' 없음")

# 4. If model has 'model' or 'transformer', check its attributes
print("\n" + "=" * 80)
print("중첩된 구조 검사:")
print("=" * 80)

for parent_name in ['model', 'transformer']:
    if hasattr(model, parent_name):
        parent = getattr(model, parent_name)
        print(f"\n'{parent_name}' 의 하위 속성들:")
        for name in dir(parent):
            if not name.startswith('_'):
                attr = getattr(parent, name, None)
                if isinstance(attr, torch.nn.Module):
                    print(f"    ✓ {parent_name}.{name}: {type(attr).__name__}")
                    # Check if this might be vision encoder
                    if 'vis' in name.lower() or 'image' in name.lower() or 'img' in name.lower():
                        print(f"      👁️  비전 관련 모듈 가능성 높음!")
                        if found_vision is None:
                            found_vision = f"{parent_name}.{name}"

# 5. Print model structure
print("\n" + "=" * 80)
print("전체 모델 구조 (간략):")
print("=" * 80)
print(model)

# 6. Recommendation
print("\n" + "=" * 80)
print("권장 사항:")
print("=" * 80)
if found_vision:
    print(f"✅ Vision 인코더로 '{found_vision}' 사용을 권장합니다.")
    print(f"\n수정할 코드:")
    print(f"  self.vision_model = model.{found_vision}")
else:
    print("⚠️  Vision 인코더를 자동으로 찾지 못했습니다.")
    print("위의 모델 구조를 확인하여 수동으로 찾아야 합니다.")

print("=" * 80)
