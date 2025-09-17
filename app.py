# app.py
import os
import io
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple
import numpy as np
import cv2
import dlib
import librosa
import soundfile as sf
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Config (edit if needed) ----------------
# Prefer container/runtime environment variables when available.
# This makes the app work both locally (Windows dev) and inside Docker (Linux paths).
import os
from pathlib import Path

# Resolve model & shape paths (env var -> container default)
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/modelav_best.pth"))
SHAPE_PRED_PATH = Path(os.getenv("SHAPE_PATH", "/app/shape_predictor_68_face_landmarks.dat"))
VOCAB_JSON = Path(os.getenv("VOCAB_JSON", "/app/vocab.json"))   # optional; model checkpoint may include 'vocab'

# Other runtime constants (leave unchanged unless you intend to change)
DESIRED_FPS = 25
FRAME_SIZE = (112, 112)  # (W,H)
SR = 50000               # sample rate used in your preprocessing (ensure match)
N_MELS = 80
WIN_MS = 25
HOP_MS = 40

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print resolved paths so logs show what the app uses (helps debugging in containers)
print(f"Resolved MODEL_PATH = {MODEL_PATH}", flush=True)
print(f"Resolved SHAPE_PRED_PATH = {SHAPE_PRED_PATH}", flush=True)
print(f"Resolved VOCAB_JSON = {VOCAB_JSON}", flush=True)


app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------------- Model definitions (must match training) ----------------
# Minimal copies of your training model pieces (VisualFrontend3D, AudioFrontend, PosEnc, ModelAV)
class VisualFrontend3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=128):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, 32, kernel_size=(3,5,5), stride=(1,2,2), padding=(1,2,2))
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, out_ch, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.bn3 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=[3,4])
        x = x.permute(0,2,1).contiguous()
        return x

class AudioFrontendCNN(nn.Module):
    def __init__(self, in_ch=1, out_ch=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=(3,3), stride=(1,2), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,2), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=(3,3), stride=(1,2), padding=(1,1))
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=3)
        x = x.permute(0,2,1).contiguous()
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :].to(x.device)

class ModelAV(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4, dropout=0.1, frontend_out=128):
        super().__init__()
        self.d_model = d_model
        self.v_frontend = VisualFrontend3D(in_ch=1, out_ch=frontend_out)
        self.a_frontend = AudioFrontendCNN(in_ch=1, out_ch=frontend_out)
        self.v_proj = nn.Linear(frontend_out, d_model)
        self.a_proj = nn.Linear(frontend_out, d_model)
        self.v_norm = nn.LayerNorm(d_model)
        self.a_norm = nn.LayerNorm(d_model)
        self.fuse_dropout = nn.Dropout(0.1)
        self.v_pos = PositionalEncoding(d_model)
        self.a_pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=1024, dropout=dropout, batch_first=True)
        self.v_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.a_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ctc_v = nn.Linear(d_model, vocab_size)
        self.ctc_a = nn.Linear(d_model, vocab_size)
        self.fuse_proj = nn.Linear(d_model, vocab_size)
    def forward(self, vis, aud, do_decoder=False, tgt_inp=None, tgt_mask=None, tgt_key_padding_mask=None):
        z_v = self.v_frontend(vis); z_v = self.v_proj(z_v); z_v = self.v_pos(z_v); z_v = self.v_enc(z_v)
        z_a = self.a_frontend(aud); z_a = self.a_proj(z_a); z_a = self.a_pos(z_a); z_a = self.a_enc(z_a)
        Tv = z_v.size(1); Ta = z_a.size(1); Tcommon = min(Tv, Ta)
        z_v_trim = z_v[:, :Tcommon, :]; z_a_trim = z_a[:, :Tcommon, :]
        logits_v = self.ctc_v(z_v_trim); logits_a = self.ctc_a(z_a_trim)
        z_v_norm = self.v_norm(z_v_trim); z_a_norm = self.a_norm(z_a_trim)
        z_fuse = 0.5 * (z_v_norm + z_a_norm); z_fuse = self.fuse_dropout(z_fuse)
        out_logits = self.fuse_proj(z_fuse)
        return out_logits, logits_v, logits_a

# ---------------- Load model and vocab ----------------
def load_model_and_vocab(model_path: Path):
    """
    Robust loader:
    - Accepts checkpoint formats: raw state_dict or dict with keys 'model_state'/'state_dict' and optional 'vocab'.
    - Tries to infer vocab (itos) from checkpoint then from vocab.json beside the checkpoint.
    - Infers vocab size V from checkpoint final linear layers when possible.
    - Constructs ModelAV with V, loads matching keys from checkpoint (strict=False) and falls back to copying only exact-shape keys.
    """
    ckpt = torch.load(str(model_path), map_location="cpu")

    # Try to extract candidate itos from checkpoint (various formats)
    itos = None
    if isinstance(ckpt, dict):
        # 1) If ckpt directly contains a 'vocab'
        if "vocab" in ckpt:
            vocab = ckpt["vocab"]
            # vocab could be list of tokens or dict
            if isinstance(vocab, list):
                itos = {i: s for i, s in enumerate(vocab)}
            elif isinstance(vocab, dict):
                # try to invert token->idx mapping or handle idx->token mapping
                try:
                    # token->idx form: {token: idx}
                    if all(isinstance(v, int) for v in vocab.values()):
                        itos = {int(i): s for s, i in vocab.items()}
                    else:
                        itos = {int(i): s for i, s in vocab.items()}
                except Exception:
                    try:
                        itos = {int(i): s for i, s in vocab.items()}
                    except Exception:
                        itos = None

    # 2) fallback to vocab.json next to checkpoint
    if itos is None:
        vfile = model_path.parent / "vocab.json"
        if vfile.exists():
            try:
                with open(vfile, "r", encoding="utf-8") as fh:
                    tokens = json.load(fh)
                itos = {i: s for i, s in enumerate(tokens)}
            except Exception:
                itos = None

    # 3) find the actual state dict inside checkpoint
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    # 4) try to infer vocab size V from state dict final layers
    V = None
    try:
        for candidate in ("ctc_v.weight", "ctc_a.weight", "fuse_proj.weight"):
            if candidate in state:
                V = int(state[candidate].shape[0])
                break
    except Exception:
        V = None

    # 5) fallback to itos length
    if V is None and itos is not None:
        V = len(itos)

    # final fallback
    if V is None:
        V = 100

    # instantiate model with inferred V
    model = ModelAV(vocab_size=V).to(DEVICE)

    # Attempt to load state_dict (first permissive)
    try:
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print("Warning loading ckpt (strict=False) error:", e)
        # try copy-only-matching-keys approach
        sd = model.state_dict()
        filtered = {}
        for k, v in state.items():
            if k in sd and sd[k].shape == v.shape:
                filtered[k] = v
        sd.update(filtered)
        model.load_state_dict(sd)

    model.eval()

    # final itos fallback if still None
    if itos is None:
        itos = {i: f"<tok{i}>" for i in range(V)}

    return model, itos


MODEL, ITOS = None, None
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model checkpoint not found: {MODEL_PATH}")
MODEL, ITOS = load_model_and_vocab(MODEL_PATH)
if ITOS is None:
    # build trivial itos if unknown (characters)
    ITOS = {0:"<blank>"}  # you should replace with actual vocab

# ---------------- Face/landmark init ----------------
if not SHAPE_PRED_PATH.exists():
    raise RuntimeError(f"Landmark predictor not found: {SHAPE_PRED_PATH}")
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(str(SHAPE_PRED_PATH))

# ---------------- Helpers ----------------
def extract_audio_from_video(video_path: str, out_wav: str):
    # use ffmpeg to extract audio reliably
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "wav", out_wav]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    # load as numpy
    audio, sr = sf.read(out_wav)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr

def compute_logmels(audio: np.ndarray, sr: int) -> np.ndarray:
    win_len = int(sr * WIN_MS / 1000)
    hop_len = int(sr * HOP_MS / 1000)
    n_fft = 1 << (win_len - 1).bit_length()
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_len, win_length=win_len,
                                       n_mels=N_MELS, fmin=0, fmax=sr//2, power=2.0)
    logm = librosa.power_to_db(S, ref=np.max).T.astype(np.float32)  # (T, n_mels)
    return logm

def extract_frames_and_crop_mouths(video_path: str, desired_fps: int = DESIRED_FPS) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or desired_fps
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # sample indices for approx desired_fps
    interval = max(1, int(round(src_fps / desired_fps)))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(frame.copy())
        idx += 1
    cap.release()
    # detect and crop mouth ROI for each frame
    crops = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)
        if len(dets) == 0:
            # fallback: center crop
            h,w = gray.shape[:2]
            cx, cy = w//2, h//2
            size = min(h,w)//3
            x1 = max(0, cx-size); y1 = max(0, cy-size); x2 = min(w, cx+size); y2 = min(h, cy+size)
            roi = gray[y1:y2, x1:x2]
        else:
            d = dets[0]
            shape = shape_predictor(gray, d)
            # mouth points indexes 48..67 in dlib 68-point
            xs = [shape.part(i).x for i in range(48,68)]
            ys = [shape.part(i).y for i in range(48,68)]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            # expand margin
            mw = x2 - x1; mh = y2 - y1
            pad_x = int(0.20 * mw) + 2
            pad_y = int(0.20 * mh) + 2
            x1 = max(0, x1 - pad_x); x2 = min(gray.shape[1], x2 + pad_x)
            y1 = max(0, y1 - pad_y); y2 = min(gray.shape[0], y2 + pad_y)
            roi = gray[y1:y2, x1:x2]
        # resize to FRAME_SIZE
        roi_resized = cv2.resize(roi, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        # convert to single channel float32 normalized [0,1]
        roi_resized = (roi_resized.astype(np.float32) / 255.0)
        crops.append(roi_resized[..., None])  # shape (H,W,1)
    if len(crops) == 0:
        raise RuntimeError("No frames extracted/cropped")
    arr = np.stack(crops, axis=0)  # (T,H,W,1)
    return arr

def prepare_tensors(vis_arr: np.ndarray, logm: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    # vis_arr: (T_vis,H,W,1) -> to (1, C, T, H, W) with C=1
    vis = vis_arr.astype(np.float32)
    T_vis = vis.shape[0]
    vis_t = torch.from_numpy(vis).permute(0,3,1,2).unsqueeze(0)  # (1, C, H, W,?) wait -> -> shape correction below
    # We need (B, C, T, H, W). vis currently (T, H, W, 1)
    vis_t = torch.from_numpy(vis).permute(3,0,1,2).unsqueeze(0).float()  # (1, C, T, H, W)
    # audio logm: (T_audio, n_mels) -> (1,1,T_audio,n_mels)
    aud = logm.astype(np.float32)
    aud_t = torch.from_numpy(aud).unsqueeze(0).unsqueeze(0).float()
    # move to device
    return vis_t.to(DEVICE), aud_t.to(DEVICE)

def greedy_ctc_decode(logits: np.ndarray, itos: dict) -> str:
    # logits: (T, V) or (1,T,V)
    if logits.ndim == 3:
        logits = logits[0]
    pred = np.argmax(logits, axis=-1).tolist()
    collapsed = []
    prev = None
    for p in pred:
        if p != prev and p != 0:
            collapsed.append(p)
        prev = p
    # map to tokens if itos available
    if itos:
        tokens = [itos.get(int(c), "") for c in collapsed]
        return " ".join([t for t in tokens if t])
    else:
        # fallback: join indices
        return " ".join([str(c) for c in collapsed])

# ---------------- Flask endpoints ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error":"no file uploaded"}), 400
    f = request.files["video"]
    tmp_dir = tempfile.mkdtemp()
    vid_path = Path(tmp_dir) / f.filename
    f.save(str(vid_path))
    try:
        # extract audio to temp wav
        wav_path = Path(tmp_dir) / "extract.wav"
        audio, sr = extract_audio_from_video(str(vid_path), str(wav_path))
        logm = compute_logmels(audio, sr)
        vis_arr = extract_frames_and_crop_mouths(str(vid_path), desired_fps=DESIRED_FPS)
        # align time: truncate to min length
        T = min(vis_arr.shape[0], logm.shape[0])
        vis_arr = vis_arr[:T]
        logm = logm[:T]
        vis_t, aud_t = prepare_tensors(vis_arr, logm)
        # run model
        with torch.no_grad():
            out_logits, logits_v, logits_a = MODEL(vis_t, aud_t)
            out_np = out_logits.detach().cpu().numpy()
        hyp = greedy_ctc_decode(out_np, ITOS)
        return jsonify({"hypothesis": hyp})
    except Exception as e:
        return jsonify({"error": str(e)})
    finally:
        # cleanup
        try:
            for p in Path(tmp_dir).glob("*"):
                p.unlink()
            Path(tmp_dir).rmdir()
        except Exception:
            pass

@app.route("/static/<path:path>")
def static_proxy(path):
    return send_from_directory("static", path)

if __name__ == "__main__":
    print("Starting server on http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
