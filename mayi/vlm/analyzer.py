from __future__ import annotations

import json
import logging
import re

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_METADATA_PROMPT = """\
Analyze this person's appearance. Return ONLY a JSON object with these fields:
{
  "gender": "male" or "female" or "unknown",
  "top_color": "main color of upper clothing",
  "top_type": "hoodie/jacket/tshirt/coat/shirt/vest/etc",
  "bottom_color": "main color of lower clothing",
  "bottom_type": "pants/shorts/skirt/etc",
  "accessories": ["hat", "bag", "glasses", ...],
  "hair": "short/long/tied/bald/etc",
  "description": "one-line overall appearance description"
}
Do not output anything except the JSON object. /no_think"""


class VLMAnalyzer:
    """Extracts structured person metadata from crop images using Qwen3-VL.

    Model loading is deferred to ``load()`` so that the heavy VLM is only
    instantiated when actually needed (``--no-vlm`` skips it entirely).
    """

    DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"

    def __init__(
        self,
        model_path: str | None = None,
        max_new_tokens: int = 256,
        device: str | None = None,
    ) -> None:
        self._model_path = model_path or self.DEFAULT_MODEL_ID
        self._max_new_tokens = max_new_tokens
        self._device = device
        self._model = None
        self._processor = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return

        from transformers import AutoModelForImageTextToText, AutoProcessor

        import os
        pretrained_ref = (
            self._model_path
            if os.path.isdir(self._model_path)
            else self._model_path
        )

        logger.info("Loading VLM model '%s' ...", pretrained_ref)

        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                pretrained_ref,
                quantization_config=bnb,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        else:
            self._model = AutoModelForImageTextToText.from_pretrained(
                pretrained_ref,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        self._processor = AutoProcessor.from_pretrained(
            pretrained_ref, trust_remote_code=True,
        )
        self._loaded = True
        logger.info("VLM model loaded.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Core inference (shared by analyzer & verifier)
    # ------------------------------------------------------------------

    def _run_inference(self, messages: list[dict]) -> str:
        from qwen_vl_utils import process_vision_info

        self.load()

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        images, videos, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self._processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            **video_kwargs,
        ).to(self._model.device)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens,
            )

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        result = self._processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    def analyze_person(self, crop_bgr: np.ndarray) -> dict | None:
        """Extract structured metadata from a person crop image.

        Returns a dict with keys matching PersonMetadata fields, or None
        on failure.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        try:
            pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": _METADATA_PROMPT},
                    ],
                }
            ]
            raw_text = self._run_inference(messages)
            logger.debug("[VLM] Raw response: %s", raw_text[:500])

            result = _parse_json(raw_text)
            if result is None:
                logger.warning(
                    "[VLM] Failed to parse JSON from response: %s",
                    raw_text[:300],
                )
            return result
        except Exception:
            logger.exception("VLM metadata extraction failed")
            return None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> VLMAnalyzer:
        return cls(
            model_path=config.get("model_path") or config.get("model"),
            max_new_tokens=config.get("max_new_tokens", 256),
        )


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Thinking-variant model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_json(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from VLM text output."""
    text = _strip_thinking(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    match = re.search(r"\{[^}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Nested JSON (e.g. "accessories": [...])
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None
