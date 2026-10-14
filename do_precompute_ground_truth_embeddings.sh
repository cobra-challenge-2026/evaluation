#!/usr/bin/env bash

# Computes MedSigLIP/FlexiCT-3D embeddings for every ground-truth case (see
# precompute_ground_truth_embeddings.py) and writes them to
# ground_truth/<CASE>/gt_embeddings.npz. evaluate.py loads these instead of
# re-embedding the (fixed) ground truth on every evaluation run.

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="cobra2026-evaluation"
GROUND_TRUTH_DIR="${SCRIPT_DIR}/ground_truth"

echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/do_build.sh"

echo "=+= Computing ground-truth embeddings"
docker run --rm \
    --gpus all \
    --platform=linux/amd64 \
    --entrypoint python \
    --volume "$GROUND_TRUTH_DIR":/opt/app/ground_truth \
    "$DOCKER_IMAGE_TAG" \
    precompute_ground_truth_embeddings.py

# The container writes gt_embeddings.npz as its own internal (non-root)
# user, so on the host they end up owned by that container UID rather than
# you. Fix that here with a throwaway root container scoped to just these
# files, so later host-side operations (do_test_run.sh's chmod, tar, git,
# rm) don't fail with a permission error.
echo "=+= Fixing ownership of the generated embeddings"
docker run --rm \
    --user 0:0 \
    --entrypoint sh \
    --volume "$GROUND_TRUTH_DIR":/opt/app/ground_truth \
    "$DOCKER_IMAGE_TAG" \
    -c "chown $(id -u):$(id -g) /opt/app/ground_truth/*/gt_embeddings.npz && chmod 644 /opt/app/ground_truth/*/gt_embeddings.npz"

echo "=+= Done. Wrote ${GROUND_TRUTH_DIR}/<CASE>/gt_embeddings.npz"
