FROM --platform=linux/amd64 docker.io/library/python:3.14-slim AS cobra2026-evaluation-amd64

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
# ----------------------------------------------------

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

RUN python -m pip install \
    --user \
    --no-color \
    --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-color \
    --no-cache-dir \
    --requirement /opt/app/requirements.txt

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY --chown=user:user ./pre_download/model_weights /opt/app/model_weights
COPY --chown=user:user helpers.py /opt/app/
COPY --chown=user:user metrics_utils.py /opt/app/
COPY --chown=user:user evaluate.py /opt/app/
COPY --chown=user:user similarity_models.py /opt/app/

# uncomment to compute embeddings inside this image Include for precompute run
# COPY --chown=user:user precompute_ground_truth_embeddings.py /opt/app/

ENV GRAND_CHALLENGE_MAX_WORKERS=1

ENV HF_HOME="/opt/app/model_weights"

ENV EMBED_DTYPE=float16

ENV MEDSIGLIP_LOCAL_DIR="/opt/app/model_weights/medsiglip"


ENTRYPOINT ["python", "evaluate.py"]
