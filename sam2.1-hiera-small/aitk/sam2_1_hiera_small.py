# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Utility helpers for preparing calibration data for the sam2.1-hiera-small model."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
from datasets import load_dataset

import torch
from transformers import AutoConfig, AutoModel

try:  # pragma: no cover - import guard for broader transformers compatibility
    from transformers import Sam2Processor  # type: ignore
except ImportError:  # pragma: no cover - Sam2Processor not available in older versions
    Sam2Processor = None

try:  # pragma: no cover - optional dependency depending on transformers version
    from transformers import AutoImageProcessor
except ImportError:  # pragma: no cover
    AutoImageProcessor = None

try:  # pragma: no cover
    from transformers import AutoProcessor
except ImportError:  # pragma: no cover
    AutoProcessor = None

from olive.data.registry import Registry

_MODEL_ID = "facebook/sam2.1-hiera-small"
_PROCESSOR = None


@Registry.register_dataset()
def load_dataset_fn(
    data_name: str = "nielsr/coco-panoptic-val2017",
    split: str = "validation",
    streaming: bool = True,
    trust_remote_code: bool = True,
    **_,
):
    return load_dataset(data_name, split=split, streaming=streaming, trust_remote_code=trust_remote_code)


def _get_processor():
    global _PROCESSOR
    if _PROCESSOR is not None:
        return _PROCESSOR

    if Sam2Processor is not None:
        _PROCESSOR = Sam2Processor.from_pretrained(_MODEL_ID, trust_remote_code=True)
        return _PROCESSOR

    if AutoImageProcessor is not None:
        _PROCESSOR = AutoImageProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
        return _PROCESSOR

    if AutoProcessor is not None:
        _PROCESSOR = AutoProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
        return _PROCESSOR

    raise ImportError(
        "transformers does not provide Sam2Processor/AutoImageProcessor/AutoProcessor; update transformers to use this recipe."
    )


class SamCalibrationDataset:
    """Dataset wrapper that serves pixel inputs for calibration."""

    def __init__(self, pixel_values: np.ndarray):
        self.pixel_values = pixel_values

    def __len__(self) -> int:  # pragma: no cover - simple proxy
        return len(self.pixel_values)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], None]:  # pragma: no cover - simple proxy
        return {"pixel_values": self.pixel_values[index]}, None


def _load_cached_dataset(cache_file: Path) -> SamCalibrationDataset:
    with np.load(cache_file) as data:
        return SamCalibrationDataset(data["pixel_values"])


def _collect_samples(dataset: Iterable[dict[str, Any]], size: int) -> SamCalibrationDataset:
    processor = _get_processor()
    samples: List[np.ndarray] = []

    for idx, item in enumerate(dataset):
        if idx >= size:
            break
        image = item["image"].convert("RGB") if hasattr(item["image"], "convert") else item["image"]
        inputs = processor(images=image, return_tensors="np")
        samples.append(inputs["pixel_values"][0].astype(np.float32))

    if not samples:
        raise ValueError("No samples were processed for calibration.")

    stacked = np.stack(samples, axis=0)
    return SamCalibrationDataset(stacked)


MODEL_EXPORT_DIR = Path("model")
ONNX_MODEL_PATH = MODEL_EXPORT_DIR / "sam2_encoder.onnx"


def export_sam2_to_onnx(
    model_id: str = _MODEL_ID,
    onnx_path: Path = ONNX_MODEL_PATH,
    opset: int = 17,
):
    """Export the SAM 2.1 model to ONNX if missing.

    This is intentionally lightweight when the file already exists
    so it can be called safely from inside Olive's model handler
    initialization path.
    """

    onnx_path = Path(onnx_path)
    if onnx_path.exists():
        return str(onnx_path)

    MODEL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, config=config, trust_remote_code=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path.as_posix(),
        input_names=["pixel_values"],
        output_names=["encoder_outputs"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "encoder_outputs": {0: "batch"},
        },
    )

    return str(onnx_path)


@Registry.register_pre_process()
def dataset_pre_process(output_data, size: int = 128, cache_key: str | None = None, **_) -> SamCalibrationDataset:
    """Prepare calibration samples and optionally cache them on disk."""

    cache_file = Path(f"./cache/data/{cache_key}_sam2_pixels_{size}.npz") if cache_key else None
    if cache_file and cache_file.exists():
        return _load_cached_dataset(cache_file)

    dataset = _collect_samples(output_data, size)

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, pixel_values=dataset.pixel_values)

    return dataset


@Registry.register_post_process()
def dataset_post_process(outputs):  # pragma: no cover - passthrough used for latency only
    return outputs
