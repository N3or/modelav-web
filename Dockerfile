# Dockerfile for ModelAV web app (CPU)
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# system deps (ffmpeg, build tools for dlib if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake wget curl ffmpeg libsndfile1 libsm6 libxrender1 libxext6 pkg-config \
    libopenblas-dev liblapack-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# copy source
COPY . /app

# upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# install CPU PyTorch wheels (ignore errors if not available)
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.0+cpu torchaudio==2.0.0+cpu torchvision==0.15.1+cpu || true

# install remaining python deps
RUN pip install -r requirements.txt

# verify dlib or try install via pip
RUN python - <<'PY' || python -m pip install dlib
try:
    import dlib
    print("dlib ok")
except Exception:
    raise SystemExit(1)
PY

# copy entrypoint (download-if-missing + start)
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# env defaults (can be overridden in Render dashboard)
ENV MODEL_PATH=/app/modelav_best.pth
ENV SHAPE_PATH=/app/shape_predictor_68_face_landmarks.dat
ENV PORT=5000

EXPOSE 5000
ENTRYPOINT ["/app/entrypoint.sh"]
