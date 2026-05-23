# ColonAI — Hugging Face Spaces / generic Docker image
#
# This Dockerfile is the way to deploy ColonAI to Hugging Face Spaces
# (their UI only offers Gradio / Docker / Static now). Set the Space SDK
# to "Docker" when creating it, point it at this repo, and HF will build
# this image automatically.
#
# Required Space env vars (set in Settings → Variables and secrets):
#   COLONAI_CHECKPOINT_HF_REPO   = Yuvraj2319/colonai-v2
#   COLONAI_CHECKPOINT_HF_FILE   = best_model.pth

FROM python:3.11-slim

# System libs needed by OpenCV / Pillow / Streamlit
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        ffmpeg curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# HF Spaces runs as the "user" UID (1000). Match that to avoid permission issues.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /home/user/app

# Install Python deps first for layer caching
COPY --chown=user:user requirements.txt ./
RUN pip install --user -r requirements.txt

# Then copy the app
COPY --chown=user:user . .

# HF Spaces expects the app on port 7860
EXPOSE 7860

# Healthcheck so HF can report when the app is actually up
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:7860/_stcore/health || exit 1

# Run Streamlit. We override .streamlit/config.toml's port/address with env
# vars above so HF can connect on its expected port (7860).
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
