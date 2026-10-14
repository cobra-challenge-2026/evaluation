"""
The following is the grand challenge simple example evaluation method example adapted for the COBRA 2026 challenge.

It is meant to run within a container. Its steps are as follows:

  1. Read the algorithm output
  2. Associate original algorithm inputs with a ground truths via predictions.json
  3. Calculate metrics by comparing the algorithm output to the ground truth
  4. Repeat for all algorithm jobs that ran for this submission
  5. Aggregate the calculated metrics
  6. Save the metrics to metrics.json

To run it locally, you can call the following bash script:

  ./do_test_run.sh

This will start the evaluation and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

The evaluation method runs inside the evaluation container on Grand Challenge. For every submitted prediction (`<CASE>_cbct.mha`) it computes, per case:
    mae, psnr      vs. the deformed planning CT (ct_def_masked.mha), inside the FOV mask
    nmi, ncc       vs. the RTK-reconstructed CBCT (cbct_rtk.mha), whole volume
    cs_medsiglip,cs_flexict cosine similarities vs. the deformed CT

The per-case results and their means ("aggregates") are written to
/output/metrics.json.

The reference embeddings are precomputed once per ground-truth case
(precompute_ground_truth_embeddings.py) and must be shipped as
<CASE>/gt_embeddings.npz inside the ground-truth tarball.

Happy programming!
"""

import glob
import json
import logging
import os
from pathlib import Path
from pprint import pformat
from statistics import mean
from typing import Dict, Optional

import SimpleITK
from helpers import run_prediction_processing, setup_logger, tree

import numpy as np

from similarity_models import SimilarityModels, load_similarity_models

from metrics_utils import (
    calculate_cc,
    calculate_mae,
    calculate_nmi,
    calculate_psnr,
)

logger = logging.getLogger("evaluate")
logger.setLevel(logging.INFO)


SCRIPT_DIR = Path(__file__).parent


def _resolve_dir(env_var: str, *candidates: Path) -> Path:
    """Pick the first existing candidate path, or an explicit override via
    env_var. Lets this run both inside the Grand Challenge container
    (absolute /input, /output, /opt/... paths) and directly with bare
    `python evaluate.py` (e.g. on a cluster, from this script's own
    directory) without editing code for each environment."""
    override = os.environ.get(env_var)
    if override:
        return Path(override)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]  # missing — fall through so the resulting error is clear


INPUT_DIRECTORY = _resolve_dir(
    "INPUT_DIRECTORY",
    Path("/input"),  # Grand Challenge container
    SCRIPT_DIR / "test" / "input",  # local/cluster dev layout (do_test_run.sh)
)
OUTPUT_DIRECTORY = _resolve_dir(
    "OUTPUT_DIRECTORY",
    Path("/output"),
    SCRIPT_DIR / "test" / "output",
)
GROUND_TRUTH_DIRECTORY = _resolve_dir(
    "GROUND_TRUTH_DIRECTORY",
    Path("/opt/ml/input/data/ground_truth"),
    SCRIPT_DIR / "ground_truth",
)
MODEL_WEIGHTS_DIR = _resolve_dir(
    "MODEL_WEIGHTS_DIR",
    Path("/opt/app/model_weights"),  # Grand Challenge container
    SCRIPT_DIR / "pre_download" / "model_weights",  # local dev layout
)

HAS_PREDICTIONS = True if (INPUT_DIRECTORY / "predictions.json").exists() else False
SIMILARITY_MODELS = None

def get_similarity_models():
    """
    Lazy loader for similarity models.
    Models are only loaded once per worker process.
    """
    global SIMILARITY_MODELS
    if SIMILARITY_MODELS is None:
        logger.info("Initializing similarity models (this should only happen once per worker)...")
        SIMILARITY_MODELS = load_similarity_models(
            device='cuda',
            flexict_checkpoint_path=str(MODEL_WEIGHTS_DIR / "flexict" / "3D_final_model.pth"),
            medsiglip_cache_dir=str(MODEL_WEIGHTS_DIR),
            )
    return SIMILARITY_MODELS
# ----------------------

def main():
    setup_logger(
        # Optionally: change this to the more verbose DEBUG
        level=logging.INFO,
    )

    log_inputs()

    metrics = {}

    if HAS_PREDICTIONS:
        predictions = read_predictions()
    else:
        print(f"Reading files directly from {INPUT_DIRECTORY}")
        predictions = [
            {"pk": Path(p).stem}
            for p in sorted(glob.glob(str(INPUT_DIRECTORY / "*.mha")))
        ]

        # Sanity check: submitted case ids must match the ground truth case ids
        case_ids = sorted(
            (Path(p).stem.split("_")[0].upper() for p in glob.glob(str(INPUT_DIRECTORY / "*.mha"))),
            reverse=True,
        )

        expected_case_ids = sorted(
            (
                    p.name.upper() 
                    for p in GROUND_TRUTH_DIRECTORY.iterdir()
                    if p.is_dir()
                    for _ in range(2) # each case has a real and a synthetic version
            ),
            reverse=True,
        )


        if expected_case_ids != case_ids:
            raise RuntimeError(
                f"Expected case ids {expected_case_ids}, but found {case_ids}"
            )


    # We now process each algorithm job for this submission
    # Note that the jobs are not in any specific order!
    # We work that out from predictions.json

    # predictions.json contains information about each job's inputs and outputs
    # along with possible timing information. There are potentially up to two times
    # available as ISO-8601 duration strings or None if not measured.
    # We advise parsing these with isodate.parse_duration, available on pypi.
    #
    # We advise caution if you are considering using these times for ranking purposes,
    # as they may not be stable for the duration of the challenge due to changes
    # in the underlying implementation or infrastructure, or be repeatable
    # due to shared hardware issues.
    #
    # `exec_duration`
    #     The duration of the execution, **if measured**. Excludes data
    #     validation, container pulling, model downloading, data downloading
    #     and data uploading times.
    #
    #     Includes model loading time, input data loading time,
    #     processing time, output data writing time and
    #     **any delays from shared hardware issues**.
    #
    # `invoke_duration`
    #     The duration of the execution, **if measured**. Excludes data
    #     validation, container pulling, model downloading, data downloading
    #     and data uploading times.
    #
    #     **Potentially excludes model loading time,
    #     depending on the users implementation**.
    #
    #     Includes input data loading time, processing time,
    #     output data writing time and
    #     **any delays from shared hardware issues**.
    #
    # One, both or neither will be set.

    # Use concurrent workers to process the predictions more efficiently
    metrics["results"] = run_prediction_processing(fn=process, predictions=predictions)

    # We have the results per prediction, we can aggregate the results and
    # generate an overall score(s) for this submission
    if metrics["results"]:
        metrics["aggregates"] = {
            "mae": mean(result["mae"] for result in metrics["results"]),
            "psnr": mean(result["psnr"] for result in metrics["results"]),
            "nmi": mean(result["nmi"] for result in metrics["results"]),
            "ncc": mean(result["ncc"] for result in metrics["results"]),
            "cs_medsiglip": mean(result["cs_medsiglip"] for result in metrics["results"]),
            "cs_flexict": mean(result["cs_flexict"] for result in metrics["results"])
        }

    # Make sure to save the metrics
    write_metrics(metrics=metrics)

    return 0


def evaluate_pair(
    ground_truth_cbct_rtk: np.ndarray,
    ground_truth_ct_def: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    similarity_models: SimilarityModels,
    ref_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:

    mae = calculate_mae(ground_truth_ct_def, prediction, mask)
    psnr = calculate_psnr(ground_truth_ct_def, prediction, mask, data_range=4095.0)
    nmi = calculate_nmi(ground_truth_cbct_rtk, prediction)
    ncc = calculate_cc(ground_truth_cbct_rtk, prediction)

    if ref_embeddings is not None:
        cs_medsiglip, cs_flexict = similarity_models.cosine_similarity_from_ref_embeddings(ref_embeddings, prediction, mask)
    else:
        cs_medsiglip, cs_flexict =  0, 0 # similarity_models.cosine_similarity(ground_truth_ct_def, prediction, mask)
    return {
        "mae": float(mae),
        "psnr": float(psnr),
        "nmi": float(nmi),
        "ncc": float(ncc),
        "cs_medsiglip": float(cs_medsiglip),
        "cs_flexict": float(cs_flexict),
    }

def process(job):
    """Processes a single job, looking at the outputs.

    job is either:
      - a full GC job dict (predictions.json phase): {"pk", "inputs", "outputs", ...}
      - a flat {"pk": "<case_id>_<...>"} entry (prediction-only phase): the
        file lives directly at INPUT_DIRECTORY / f"{pk}.mha"
    """
    report = "Processing Job:\n"
    report += pformat(job)
    report += "\n"

    # Firstly, find the location of the results
    if HAS_PREDICTIONS:
        location_cbct_image = get_file_location(
            job_pk=job["pk"],
            values=job["outputs"],
            slug="cbct-image",
        )

        # Secondly, read the results
        prediction_cbct_image = load_image_file_as_array(
            location=location_cbct_image,
        )

        # Thirdly, retrieve the input file name to match it with your ground truth
        image_name_cbct_projections = get_image_name(
            values=job["inputs"],
            slug="cbct-projections",
        )
    else:
        pk = job["pk"]
        location_cbct_image = INPUT_DIRECTORY / f"{pk}.mha"

        prediction_cbct_image_sitk = SimpleITK.ReadImage(str(location_cbct_image))
        prediction_cbct_image = SimpleITK.GetArrayFromImage(prediction_cbct_image_sitk)

        image_name_cbct_projections = pk

    # Ground-truth case folders are named by institution letter + 3-digit id
    # (e.g. "A009", "B000", uppercase).  
    case_id = image_name_cbct_projections.split("_")[0].upper()

    # Fourthly, load ground truth
    case_dir = GROUND_TRUTH_DIRECTORY / case_id

    ground_truth_ct_def_sitk = SimpleITK.ReadImage(str(case_dir / "ct_def_masked.mha"))
    ground_truth_ct_def = SimpleITK.GetArrayFromImage(ground_truth_ct_def_sitk)

    ground_truth_cbct_rtk_sitk = SimpleITK.ReadImage(str(case_dir / "cbct_rtk.mha"))
    ground_truth_cbct_rtk = SimpleITK.GetArrayFromImage(ground_truth_cbct_rtk_sitk)

    if prediction_cbct_image.shape != ground_truth_ct_def.shape:
        raise RuntimeError(
            f"Shape mismatch for case {case_id}: algorithm output has shape "
            f"{prediction_cbct_image.shape}, ground truth ct def has shape {ground_truth_ct_def.shape}"
        )

    mask_sitk = SimpleITK.ReadImage(str(case_dir / "fov_cbct.mha"))
    mask = SimpleITK.GetArrayFromImage(mask_sitk) > 0

    similarity_models = get_similarity_models()

    # Ground truth is fixed across every submission, so if its MedSigLIP/
    # FlexiCT embeddings were precomputed (see precompute_ground_truth_embeddings.py)
    # and shipped alongside it in the ground-truth tarball, reuse them here
    # instead of re-embedding the reference volume on every evaluation run.
    ref_embeddings_path = case_dir / "gt_embeddings.npz"
    if ref_embeddings_path.exists():
        ref_embeddings = dict(np.load(ref_embeddings_path))
    else:
        logger.warning(
            "No precomputed ground-truth embeddings at %s; embedding the "
            "reference volume on this run instead (see "
            "precompute_ground_truth_embeddings.py).",
            ref_embeddings_path,
        )
        ref_embeddings = None

    metrics = evaluate_pair(
        ground_truth_ct_def=ground_truth_ct_def,
        ground_truth_cbct_rtk=ground_truth_cbct_rtk,
        prediction=prediction_cbct_image,
        mask=mask,
        similarity_models=similarity_models,
        ref_embeddings=ref_embeddings,
    )

    report += f"\nMetrics calculated: {metrics}"

    logger.info("report:")
    logger.info(report)

    return {
        "mae": metrics["mae"],
        "psnr": metrics["psnr"],
        "nmi": metrics["nmi"],
        "ncc": metrics["ncc"],
        "cs_medsiglip": metrics["cs_medsiglip"],
        "cs_flexict": metrics["cs_flexict"],
    }


def log_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    logger.info("Input Files:")
    for line in tree(INPUT_DIRECTORY):
        logger.info(line)


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    return load_json_file(location=INPUT_DIRECTORY / "predictions.json")


def get_interface_key(job):
    # Each interface has a unique key that is the set of socket slugs given as input
    socket_slugs = [sv["socket"]["slug"] for sv in job["inputs"]]
    return tuple(sorted(socket_slugs))


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["socket"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["socket"]["slug"] == slug:
            return value["socket"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob.glob(str(location / "*.tif"))
        + glob.glob(str(location / "*.tiff"))
        + glob.glob(str(location / "*.mha"))
    )
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    write_json_file(location=OUTPUT_DIRECTORY / "metrics.json", content=metrics)


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


if __name__ == "__main__":
    raise SystemExit(main())
