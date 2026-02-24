#!/usr/bin/env python3
"""Test VLM-based Re-ID verification with two cropped person images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from mayi.vlm.analyzer import VLMAnalyzer
from mayi.vlm.verifier import VLMVerifier


def load_image(path: str) -> np.ndarray | None:
    """Load image as BGR numpy array."""
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Failed to load image: {path}")
        return None
    return img


def show_images_side_by_side(img1: np.ndarray, img2: np.ndarray, result: dict):
    """Display two images side by side with result overlay."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Match heights
    max_h = max(h1, h2)
    if h1 < max_h:
        scale = max_h / h1
        img1 = cv2.resize(img1, (int(w1 * scale), max_h))
    if h2 < max_h:
        scale = max_h / h2
        img2 = cv2.resize(img2, (int(w2 * scale), max_h))

    # Concatenate horizontally
    combined = np.hstack([img1, img2])

    # Add result text
    is_same = result["is_same"]
    confidence = result["confidence"]
    reason = result.get("reason", "")

    text = f"Same: {is_same} | Conf: {confidence:.2f}"
    color = (0, 255, 0) if is_same else (0, 0, 255)

    cv2.putText(
        combined, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
    )

    if reason:
        cv2.putText(
            combined, f"Reason: {reason}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )

    cv2.imshow("VLM RE-ID Test", combined)
    print("\n💡 Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Test VLM Re-ID with two person crop images",
    )
    parser.add_argument(
        "image1",
        type=str,
        help="Path to first person crop image",
    )
    parser.add_argument(
        "image2",
        type=str,
        help="Path to second person crop image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="VLM model path (default: Qwen/Qwen3-VL-8B-Thinking)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max new tokens for VLM generation (default: 64)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display images (text output only)",
    )

    args = parser.parse_args()

    # Load images
    print(f"📂 Loading images...")
    img1 = load_image(args.image1)
    img2 = load_image(args.image2)

    if img1 is None or img2 is None:
        sys.exit(1)

    print(f"✅ Image 1: {img1.shape} (H×W×C)")
    print(f"✅ Image 2: {img2.shape} (H×W×C)")

    # Initialize VLM
    print(f"\n🤖 Initializing VLM...")
    analyzer = VLMAnalyzer(
        model_path=args.model,
        max_new_tokens=256,
    )
    verifier = VLMVerifier(
        analyzer=analyzer,
        max_retries=2,
        max_new_tokens=args.max_tokens,
    )

    # Load model
    print(f"⏳ Loading model (this may take a while)...")
    analyzer.load()

    # Run verification
    print(f"\n🔍 Running VLM Re-ID verification...")
    is_same, confidence = verifier.verify(img1, img2)

    # Print results
    print("\n" + "=" * 60)
    print("📊 RESULT")
    print("=" * 60)
    print(f"Same Person:  {'✅ YES' if is_same else '❌ NO'}")
    print(f"Confidence:   {confidence:.2f}")
    print("=" * 60)

    # Show images
    if not args.no_show:
        result = {
            "is_same": is_same,
            "confidence": confidence,
        }
        show_images_side_by_side(img1, img2, result)


if __name__ == "__main__":
    main()
