#!/usr/bin/env python3
"""Inspect Qwen3VLVisionModel.forward() signature."""

import inspect
import torch
from transformers import AutoModelForImageTextToText

print("=" * 80)
print("Qwen3VLVisionModel.forward() 시그니처 검사")
print("=" * 80)

# Load model
print("\n모델 로딩 중...")
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-8B-Thinking",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

vision_model = model.model.visual

print(f"\n✅ Vision 모델 타입: {type(vision_model).__name__}")

# Get forward signature
print("\n" + "=" * 80)
print("forward() 메서드 시그니처:")
print("=" * 80)

forward_signature = inspect.signature(vision_model.forward)
print(f"\n{forward_signature}")

print("\n파라미터 상세:")
for param_name, param in forward_signature.parameters.items():
    default = param.default
    if default == inspect.Parameter.empty:
        default_str = "Required (no default)"
    else:
        default_str = f"Default: {default}"

    annotation = param.annotation
    if annotation == inspect.Parameter.empty:
        type_str = "Any"
    else:
        type_str = str(annotation)

    print(f"  - {param_name}: {type_str}")
    print(f"    {default_str}")

# Try to get docstring
print("\n" + "=" * 80)
print("forward() 독스트링:")
print("=" * 80)
if vision_model.forward.__doc__:
    print(vision_model.forward.__doc__)
else:
    print("독스트링 없음")

# Check if there's example usage in the model
print("\n" + "=" * 80)
print("모델 사용 예제 찾기:")
print("=" * 80)

# Try with processor
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen3-VL-8B-Thinking",
    trust_remote_code=True,
)

print("\n✅ Processor 로딩 완료")
print(f"Processor 타입: {type(processor).__name__}")

# Create a dummy image to see what the processor returns
from PIL import Image
import numpy as np

dummy_image = Image.fromarray(
    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
)

print("\n더미 이미지로 processor 테스트...")
inputs = processor(
    images=dummy_image,
    return_tensors="pt",
)

print("\nProcessor가 반환한 키들:")
for key, value in inputs.items():
    if isinstance(value, torch.Tensor):
        print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  - {key}: {type(value)}")

# Check if grid_thw is in the inputs
if 'image_grid_thw' in inputs or 'grid_thw' in inputs:
    grid_key = 'image_grid_thw' if 'image_grid_thw' in inputs else 'grid_thw'
    print(f"\n✅ '{grid_key}' 발견!")
    print(f"   값: {inputs[grid_key]}")
    print(f"   Shape: {inputs[grid_key].shape if isinstance(inputs[grid_key], torch.Tensor) else 'N/A'}")

print("\n" + "=" * 80)
print("권장 사항:")
print("=" * 80)
print("VisionEncoderWrapper.forward()에 grid_thw 파라미터를 추가해야 합니다.")
print("Processor가 반환하는 값을 확인하여 올바른 grid_thw를 생성하세요.")
print("=" * 80)
