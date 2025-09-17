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

# install CPU PyTorch wheels
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.0+cpu torchaudio==2.0.0+cpu torchvision==0.15.1+cpu || true

# install remaining python deps (requirements.txt should be present)
RUN pip install -r requirements.txt

# ensure dlib available (attempt import to detect wheel; pip install if missing)
RUN python - <<'PY' || python -m pip install dlib
try:
    import dlib
    print("dlib ok")
except Exception as e:
    raise SystemExit(1)
PY

# environment defaults (can be overridden at runtime/build)
ENV MODEL_PATH=/app/modelav_best.pth
ENV SHAPE_PATH=/app/shape_predictor_68_face_landmarks.dat

# accept build args to download model files during build
ARG MODEL_URL
ARG SHAPE_URL
RUN if [ -n "$MODEL_URL" ] ; then curl -L -o $MODEL_PATH "$MODEL_URL" ; fi
RUN if [ -n "$SHAPE_URL" ] ; then curl -L -o $SHAPE_PATH "$SHAPE_URL" ; fi

EXPOSE 5000
CMD ["waitress-serve", "--port=5000", "app:app"]
