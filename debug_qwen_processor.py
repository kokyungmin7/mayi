#!/usr/bin/env python3
"""Debug script to understand Qwen3-VL processor output format.

This helps us understand the exact format of pixel_values and grid_thw
that the vision model expects.
"""

import logging
import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Investigate Qwen3-VL processor and vision model."""

    logger.info("="*70)
    logger.info("Qwen3-VL Processor & Vision Model Investigation")
    logger.info("="*70)

    # Load processor and model
    model_name = "Qwen/Qwen3-VL-8B-Thinking"

    logger.info(f"\nLoading processor: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    logger.info(f"Loading model (FP16, GPU)...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Create test image (224x224)
    logger.info("\n" + "="*70)
    logger.info("Test 1: 224x224 Image")
    logger.info("="*70)

    test_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    test_prompt = "Describe this image."

    # Process image with processor
    logger.info("\nProcessing image with Qwen3-VL processor...")

    # Qwen3-VL uses conversation format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": test_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[test_image],
        return_tensors="pt",
    )

    # Move to GPU
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    logger.info("\nProcessor outputs:")
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            logger.info(f"  {key}: {type(value)}")

    # Check if grid_thw exists
    if "grid_thw" in inputs:
        logger.info(f"\nFound grid_thw!")
        logger.info(f"  Value: {inputs['grid_thw']}")
        logger.info(f"  Shape: {inputs['grid_thw'].shape}")
    elif "image_grid_thw" in inputs:
        logger.info(f"\nFound image_grid_thw!")
        logger.info(f"  Value: {inputs['image_grid_thw'].tolist()}")
        logger.info(f"  Shape: {inputs['image_grid_thw'].shape}")
    else:
        logger.info(f"\nNo 'grid_thw' or 'image_grid_thw' in processor output")
        logger.info(f"  Available keys: {list(inputs.keys())}")

    # Check pixel_values
    if "pixel_values" in inputs:
        pv = inputs["pixel_values"]
        logger.info(f"\npixel_values analysis:")
        logger.info(f"  Shape: {pv.shape}")
        logger.info(f"  Dtype: {pv.dtype}")
        logger.info(f"  Min/Max: {pv.min():.3f} / {pv.max():.3f}")
        logger.info(f"  Device: {pv.device}")

        # Try to understand the format
        if len(pv.shape) == 4:
            logger.info(f"  Format: (batch={pv.shape[0]}, channels={pv.shape[1]}, "
                       f"height={pv.shape[2]}, width={pv.shape[3]})")
        elif len(pv.shape) == 3:
            logger.info(f"  Format: (batch={pv.shape[0]}, seq_len={pv.shape[1]}, "
                       f"features={pv.shape[2]})")

    # Inspect vision model
    logger.info("\n" + "="*70)
    logger.info("Vision Model Inspection")
    logger.info("="*70)

    vision_model = model.model.visual
    logger.info(f"\nVision model type: {type(vision_model)}")
    logger.info(f"Vision model config:")

    if hasattr(vision_model, "config"):
        config = vision_model.config
        for attr in ["hidden_size", "patch_size", "temporal_patch_size",
                     "spatial_merge_size", "temporal_merge_size"]:
            if hasattr(config, attr):
                logger.info(f"  {attr}: {getattr(config, attr)}")

    # Check forward signature
    import inspect
    sig = inspect.signature(vision_model.forward)
    logger.info(f"\nVision model forward() signature:")
    logger.info(f"  Parameters: {list(sig.parameters.keys())}")

    # Try forward pass with processor outputs
    logger.info("\n" + "="*70)
    logger.info("Test Forward Pass")
    logger.info("="*70)

    try:
        with torch.no_grad():
            # Extract relevant inputs for vision model
            vision_inputs = {}
            if "pixel_values" in inputs:
                vision_inputs["hidden_states"] = inputs["pixel_values"]
            if "grid_thw" in inputs:
                vision_inputs["grid_thw"] = inputs["grid_thw"]

            logger.info(f"\nCalling vision_model.forward() with:")
            for k, v in vision_inputs.items():
                if isinstance(v, torch.Tensor):
                    logger.info(f"  {k}: {v.shape}")

            vision_output = vision_model(**vision_inputs)

            if hasattr(vision_output, "last_hidden_state"):
                output_shape = vision_output.last_hidden_state.shape
            elif isinstance(vision_output, torch.Tensor):
                output_shape = vision_output.shape
            else:
                output_shape = "unknown"

            logger.info(f"\n✅ Vision model forward pass successful!")
            logger.info(f"  Output shape: {output_shape}")

    except Exception as e:
        logger.error(f"\n❌ Vision model forward pass failed: {e}")
        import traceback
        traceback.print_exc()

    # Test different image sizes
    logger.info("\n" + "="*70)
    logger.info("Test 2: Different Image Sizes")
    logger.info("="*70)

    for size in [224, 384, 512]:
        logger.info(f"\nTesting {size}x{size} image:")

        test_img = Image.fromarray(
            np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_img},
                    {"type": "text", "text": "Test"},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[test_img], return_tensors="pt")

        if "pixel_values" in inputs:
            logger.info(f"  pixel_values shape: {inputs['pixel_values'].shape}")
        if "grid_thw" in inputs:
            logger.info(f"  grid_thw: {inputs['grid_thw'].tolist()}")

    logger.info("\n" + "="*70)
    logger.info("Investigation Complete")
    logger.info("="*70)
    logger.info("\nKey findings will help us fix the ONNX export!")


if __name__ == "__main__":
    main()
