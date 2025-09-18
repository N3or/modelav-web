#!/bin/sh
set -e

echo "ENTRYPOINT: ensure model files"

# GitHub release assets are public — if MODEL_URL is set we will download
if [ ! -f /app/modelav_best.pth ] && [ -n "$MODEL_URL" ]; then
  echo "Downloading model from: $MODEL_URL"
  if [ -n "$MODEL_AUTH_TOKEN" ]; then
    # if private asset, use token (set MODEL_AUTH_TOKEN in Render)
    curl -L -H "Authorization: token $MODEL_AUTH_TOKEN" -o /app/modelav_best.pth "$MODEL_URL"
  else
    curl -L -o /app/modelav_best.pth "$MODEL_URL"
  fi
fi

if [ ! -f /app/shape_predictor_68_face_landmarks.dat ] && [ -n "$SHAPE_URL" ]; then
  echo "Downloading shape predictor from: $SHAPE_URL"
  if [ -n "$MODEL_AUTH_TOKEN" ]; then
    curl -L -H "Authorization: token $MODEL_AUTH_TOKEN" -o /app/shape_predictor_68_face_landmarks.dat "$SHAPE_URL"
  else
    curl -L -o /app/shape_predictor_68_face_landmarks.dat "$SHAPE_URL"
  fi
fi

echo "Starting server..."
# Use PORT if provided by Render, otherwise default to 5000
exec waitress-serve --port=${PORT:-5000} app:app
