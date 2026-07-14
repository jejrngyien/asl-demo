"""
ASL fingerspelling demo (Streamlit).

Classifies a single American Sign Language fingerspelling handshape
(A-Z + del/space/nothing) from a webcam snapshot or an uploaded image,
using the static C3D model from the ASL Recognition with 3D CNNs project.

The model weights are downloaded once from a GitHub Release (see MODEL_URL)
so the git repo stays small; a local `model.pt` is used if present.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import streamlit as st

from models import build_model

# --- Where to get the weights ---------------------------------------------
# After you create a GitHub Release with model.pt attached, put its URL here
# (or set it as a Streamlit secret / env var named MODEL_URL). Example:
#   https://github.com/USERNAME/asl-demo/releases/download/v1.0/model.pt
MODEL_URL = (
    st.secrets.get("MODEL_URL", "") if hasattr(st, "secrets") else ""
) or os.environ.get("MODEL_URL", "") or "PUT_YOUR_GITHUB_RELEASE_URL_HERE"
LOCAL_MODEL = "model.pt"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- Preprocessing (mirrors the main repo) --------------------
def resize_center_crop(rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = min(h, w)
    y1 = max(0, (h - s) // 2)
    x1 = max(0, (w - s) // 2)
    return cv2.resize(rgb[y1:y1 + s, x1:x1 + s], (size, size), interpolation=cv2.INTER_LINEAR)


def bbox_from_landmarks(lms, W, H, pad=0.35):
    xs = [int(p.x * W) for p in lms]
    ys = [int(p.y * H) for p in lms]
    x1, x2 = max(min(xs), 0), min(max(xs), W - 1)
    y1, y2 = max(min(ys), 0), min(max(ys), H - 1)
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2
    side = int(max(w, h) * (1.0 + pad))
    x1n, y1n = max(cx - side // 2, 0), max(cy - side // 2, 0)
    return x1n, y1n, min(x1n + side, W - 1), min(y1n + side, H - 1)


def to_tensor_normalized(rgb_uint8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(rgb_uint8).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    return (x - mean) / std


# -------------------- Cached model / detector loading --------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    path = LOCAL_MODEL
    if not os.path.exists(path):
        if not MODEL_URL or MODEL_URL == "PUT_YOUR_GITHUB_RELEASE_URL_HERE":
            st.error("No model found. Set MODEL_URL to your GitHub Release URL, "
                     "or place model.pt next to this app.")
            st.stop()
        torch.hub.download_url_to_file(MODEL_URL, path, progress=False)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    classes = [str(c) for c in ckpt["classes"]]
    arch = ckpt.get("model", "c3d")
    img_size = int(ckpt.get("img_size", 112))
    state = ckpt.get("model_state", ckpt)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model = build_model(arch, num_classes=len(classes))
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model, classes, img_size


@st.cache_resource(show_spinner=False)
def load_hands():
    try:
        import mediapipe as mp
        if hasattr(mp, "solutions"):
            return mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                            model_complexity=1, min_detection_confidence=0.5)
    except Exception:
        pass
    return None


def crop_hand(rgb: np.ndarray, hands):
    H, W = rgb.shape[:2]
    if hands is not None:
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            x1, y1, x2, y2 = bbox_from_landmarks(res.multi_hand_landmarks[0].landmark, W, H)
            if (x2 - x1) > 4 and (y2 - y1) > 4:
                return rgb[y1:y2, x1:x2].copy(), True
    side = int(min(H, W) * 0.7)
    cx, cy = W // 2, H // 2
    x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
    return rgb[y1:y1 + side, x1:x1 + side].copy(), False


@torch.no_grad()
def predict(rgb: np.ndarray, model, classes, img_size, hands):
    crop, used_mp = crop_hand(rgb, hands)
    if crop.size == 0:
        crop = rgb
    crop = resize_center_crop(crop, img_size)
    x = to_tensor_normalized(crop).unsqueeze(0).unsqueeze(2).to(DEVICE)  # [1,C,1,H,W]
    probs = F.softmax(model(x), dim=1)[0].cpu().numpy().astype(float)
    return probs, crop, used_mp


# -------------------- UI --------------------
st.set_page_config(page_title="ASL Fingerspelling Demo", page_icon="🤟")
st.title("🤟 ASL Fingerspelling Recognition")
st.markdown(
    "Show an **American Sign Language fingerspelling** handshape to your webcam "
    "(or upload a photo) and the C3D model predicts the letter "
    "(**A–Z + del/space/nothing**). A MediaPipe hand detector crops your hand first.\n\n"
    "Part of the [ASL Recognition with 3D CNNs](https://github.com/jejrngyien/asl) project · "
    "~84% Top-1 on the Kaggle ASL Alphabet."
)

model, classes, img_size = load_model()
hands = load_hands()

tab_cam, tab_up = st.tabs(["📷 Webcam", "🖼️ Upload"])
image = None
with tab_cam:
    shot = st.camera_input("Take a picture of your handshape")
    if shot is not None:
        image = np.array(Image.open(shot).convert("RGB"))
with tab_up:
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if up is not None:
        image = np.array(Image.open(up).convert("RGB"))

if image is not None:
    probs, crop, used_mp = predict(image, model, classes, img_size, hands)
    order = np.argsort(-probs)[:5]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(crop, caption="What the model sees (112×112 crop)", width=224)
        if not used_mp:
            st.warning("No hand detected — used a centered crop. "
                       "Try a clearer handshape on a plain background.")
    with col2:
        top = classes[order[0]]
        st.metric("Prediction", top, f"{probs[order[0]] * 100:.1f}%")
        st.caption("Top-5")
        st.bar_chart({classes[i]: float(probs[i]) for i in order})

st.divider()
st.caption("Tip: letters Y, C, L, B, V are the easiest. Look up an ASL alphabet chart for the handshapes.")
