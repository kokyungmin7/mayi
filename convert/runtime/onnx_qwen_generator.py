"""ONNX Runtime wrapper for Qwen3-VL generation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor

logger = logging.getLogger(__name__)


class ONNXQwenGenerator:
    """Qwen3-VL ONNX inference with autoregressive generation.

    This class loads the vision encoder and text decoder ONNX models
    and implements the generation loop in Python.
    """

    def __init__(
        self,
        vision_encoder_path: str | Path,
        text_decoder_path: str | Path,
        processor_name: str = "Qwen/Qwen3-VL-8B-Thinking",
        providers: list[str] | None = None,
    ):
        """Initialize ONNX generator.

        Args:
            vision_encoder_path: Path to vision encoder ONNX
            text_decoder_path: Path to text decoder ONNX
            processor_name: HuggingFace processor name
            providers: ONNX Runtime providers (e.g., ["CUDAExecutionProvider"])
        """
        if providers is None:
            providers = ["CPUExecutionProvider"]

        logger.info("Loading vision encoder: %s", vision_encoder_path)
        self.vision_session = ort.InferenceSession(
            str(vision_encoder_path),
            providers=providers,
        )

        logger.info("Loading text decoder: %s", text_decoder_path)
        self.decoder_session = ort.InferenceSession(
            str(text_decoder_path),
            providers=providers,
        )

        logger.info("Loading processor: %s", processor_name)
        self.processor = AutoProcessor.from_pretrained(
            processor_name,
            trust_remote_code=True,
        )

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode image to vision embeddings using ONNX.

        Args:
            image: PIL Image

        Returns:
            vision_embeds: [1, num_patches, hidden_dim]
        """
        # Preprocess image (using HuggingFace processor)
        inputs = self.processor(images=image, return_tensors="np")
        pixel_values = inputs["pixel_values"]

        # Run vision encoder
        vision_embeds = self.vision_session.run(
            ["vision_embeds"],
            {"pixel_values": pixel_values},
        )[0]

        return vision_embeds

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> str:
        """Generate text from image and prompt.

        Args:
            image: PIL Image
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            generated_text: Generated response
        """
        # Encode image
        vision_embeds = self.encode_image(image)

        # Tokenize prompt
        text_inputs = self.processor(
            text=prompt,
            return_tensors="np",
        )
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        # Autoregressive generation loop
        generated_ids = input_ids.copy()

        for _ in range(max_new_tokens):
            # Run decoder
            logits = self.decoder_session.run(
                ["logits"],
                {
                    "input_ids": generated_ids,
                    "attention_mask": attention_mask,
                    "vision_embeds": vision_embeds,
                },
            )[0]

            # Get next token (greedy decoding)
            next_token_logits = logits[:, -1, :]
            next_token_id = np.argmax(next_token_logits, axis=-1, keepdims=True)

            # Append to sequence
            generated_ids = np.concatenate([generated_ids, next_token_id], axis=1)
            attention_mask = np.ones_like(generated_ids)

            # Check for EOS token
            if next_token_id[0, 0] == self.processor.tokenizer.eos_token_id:
                break

        # Decode
        generated_text = self.processor.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        return generated_text
