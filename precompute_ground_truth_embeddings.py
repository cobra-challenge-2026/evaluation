#!/usr/bin/env python
"""Precomputes MedSigLIP/FlexiCT-3D embeddings for every ground-truth case.

The ground truth (ground_truth/<CASE>/ct_def_masked.mha) is fixed across
every algorithm submission, but evaluate.py used to re-embed it on every
single evaluation run alongside the (actually variable) prediction. Run this
once, locally, whenever the ground_truth/ directory changes. evaluate.py
loads them instead of recomputing when present.
"""
import os
from pathlib import Path

import numpy as np
import SimpleITK

from similarity_models import load_similarity_models

GROUND_TRUTH_DIR = Path(__file__).parent / "ground_truth"
GT_IMAGE_FILENAME = "ct_def_masked.mha"
FOV_MASK_FILENAME = "fov_cbct.mha"
EMBEDDINGS_FILENAME = "gt_embeddings.npz"

# /opt/app/model_weights in the container
# MODEL_WEIGHTS_DIR=<...>/pre_download/model_weights.
MODEL_WEIGHTS_DIR = Path(os.environ.get("MODEL_WEIGHTS_DIR", "/opt/app/model_weights"))


def main() -> None:
    case_dirs = sorted(
        p for p in GROUND_TRUTH_DIR.iterdir() if p.is_dir() and (p / GT_IMAGE_FILENAME).exists()
    )
    print(f"Found {len(case_dirs)} ground-truth case(s) under {GROUND_TRUTH_DIR}")

    models = load_similarity_models(
        flexict_checkpoint_path=str(MODEL_WEIGHTS_DIR / "flexict" / "3D_final_model.pth"),
        medsiglip_cache_dir=str(MODEL_WEIGHTS_DIR / "medsiglip"),
        device="cuda",
    )

    for case_dir in case_dirs:
        image_path = case_dir / GT_IMAGE_FILENAME
        volume = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(image_path)))

        mask_path = case_dir / FOV_MASK_FILENAME
        mask = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(mask_path))) > 0

        embeddings = models.embed_reference(volume, mask)

        out_path = case_dir / EMBEDDINGS_FILENAME
        np.savez(out_path, **embeddings)
        print(f"  {case_dir.name}: wrote {out_path}")


if __name__ == "__main__":
    main()
