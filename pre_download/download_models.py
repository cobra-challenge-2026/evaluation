#!/usr/bin/env python
# coding: utf-8
"""Downloads the model weights used by the evaluation into ./model_weights/.

Layout produced (this is what the Dockerfile copies to /opt/app/model_weights):

  model_weights/flexict/3D_final_model.pth   FlexiCT-3D checkpoint (Google Drive)
  model_weights/medsiglip/                   google/medsiglip-448 (Hugging Face, gated)

google/medsiglip-448 is a GATED model. Before running this script:
  1. Accept the license at https://huggingface.co/google/medsiglip-448
  2. Authenticate: `huggingface-cli login` (or set HF_TOKEN).
Without that the download fails with a 401/403.

Run from the pre_download/ directory:  python download_models.py
"""
import os

from huggingface_hub import snapshot_download

# Must match similarity_models.MEDSIGLIP_MODEL_ID.
MEDSIGLIP_MODEL_ID = "google/medsiglip-448"
MEDSIGLIP_DIR = "./model_weights/medsiglip"

# See: https://github.com/ricklisz/FlexiCT
FLEXICT_3D_GDRIVE_ID = "1Y599UnO4ab-vJCVhn5eOI8fECoHVWRyv"
FLEXICT_3D_WEIGHTS_DIR = "./model_weights/flexict"
FLEXICT_3D_WEIGHTS_PATH = f"{FLEXICT_3D_WEIGHTS_DIR}/3D_final_model.pth"


def pre_download():
    os.makedirs(FLEXICT_3D_WEIGHTS_DIR, exist_ok=True)

    print("Downloading FlexiCT-3D weights...")
    import gdown

    gdown.download(
        id=FLEXICT_3D_GDRIVE_ID,
        output=FLEXICT_3D_WEIGHTS_PATH,
        quiet=False,
    )

    # A flat local folder (not the Hugging Face cache layout): the container
    # points MEDSIGLIP_LOCAL_DIR at it and runs fully offline.
    print(f"Downloading MedSigLIP weights ({MEDSIGLIP_MODEL_ID})...")
    snapshot_download(
        MEDSIGLIP_MODEL_ID,
        local_dir=MEDSIGLIP_DIR,
        allow_patterns=["*.json", "*.safetensors*"],
    )

    print("All models successfully downloaded to ./model_weights/ !")


if __name__ == "__main__":
    pre_download()
