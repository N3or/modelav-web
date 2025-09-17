# Dockerfile for ModelAV web app (CPU)
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# system deps (ffmpeg, build tools for dlib if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake wget curl ffmpeg libsndfile1 libsm6 libxrender1 libxext6 pkg-config \
    libopenblas-dev liblapack-dev bzip2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# copy source code
COPY . /app

# download pretrained model + shape predictor directly
RUN wget -q -O /app/modelav_best.pth https://github.com/N3or/modelav-web/releases/download/v1.0/modelav_best.pth && \
    wget -q -O /app/shape_predictor_68_face_landmarks.dat https://github.com/N3or/modelav-web/releases/download/v1.0/shape_predictor_68_face_landmarks.dat

# upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# install CPU PyTorch wheels
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.0.0+cpu" "torchaudio==2.0.0+cpu" "torchvision==0.15.1+cpu" || true

# install remaining python deps
RUN pip install -r requirements.txt

# ensure waitress is available for CMD
RUN pip install waitress

# ensure dlib available
RUN python - <<'PY' || python -m pip install dlib
try:
    import dlib
    print("dlib ok")
except Exception:
    raise SystemExit(1)
PY

# environment defaults (point to downloaded files)
ENV MODEL_PATH=/app/modelav_best.pth
ENV SHAPE_PATH=/app/shape_predictor_68_face_landmarks.dat

EXPOSE 5000
CMD ["waitress-serve", "--port=5000", "app:app"]
