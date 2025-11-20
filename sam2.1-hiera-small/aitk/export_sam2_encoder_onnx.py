from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel  # swap to SAM2-specific class if available

MODEL_ID = "facebook/sam2.1-hiera-small"
MODEL_EXPORT_DIR = Path(__file__).parent / "model"
ONNX_PATH = MODEL_EXPORT_DIR / "sam2_encoder.onnx"


def main():
    MODEL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    # TODO: if there is a dedicated encoder class, replace AutoModel with it
    model = AutoModel.from_pretrained(MODEL_ID, config=config, trust_remote_code=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_input,),
        ONNX_PATH.as_posix(),
        input_names=["pixel_values"],
        output_names=["encoder_outputs"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "encoder_outputs": {0: "batch"},
        },
    )
    print(f"Exported encoder ONNX to {ONNX_PATH}")


if __name__ == "__main__":
    main()