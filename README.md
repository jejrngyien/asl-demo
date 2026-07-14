# 🤟 ASL Fingerspelling Recognition — Live Demo

Interactive **Streamlit** demo of a **C3D** model that recognizes **American Sign
Language fingerspelling** (A–Z + del/space/nothing) from a single image. Show a
handshape to your webcam or upload a photo — the app crops your hand with
**MediaPipe** and predicts the letter.

Deployable front-end for the
**[ASL Recognition with 3D CNNs](https://github.com/jejrngyien/asl)** project
(training pipeline, R(2+1)D comparison, and word-level WLASL experiments live there).

## How the model is loaded

The weights (~330 MB) are **not** committed. They are downloaded once from a
**GitHub Release** the first time the app starts. Set the URL via the `MODEL_URL`
constant in [`streamlit_app.py`](streamlit_app.py) or as a Streamlit secret, e.g.:

```
https://github.com/USERNAME/asl-demo/releases/download/v1.0/model.pt
```

A local `model.pt` next to the app is used if present (handy for local runs).

## Run locally

```bash
pip install -r requirements.txt
# place the checkpoint next to the app as model.pt (or set MODEL_URL)
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud (free)

1. Push this folder to a **GitHub** repo.
2. Create a **Release** on that repo and attach `model.pt` as an asset.
3. Put the release download URL into `MODEL_URL` (in `streamlit_app.py` or as a secret).
4. Go to **share.streamlit.io** → **New app** → pick the repo and
   `streamlit_app.py` → **Deploy**.

## Notes

- **Model:** static C3D, ~84% Top-1 on the Kaggle ASL Alphabet test set.
- **Input:** the model sees a 112×112 hand crop (shown in the app for transparency).
- **Tip:** letters **Y, C, L, B, V** are the easiest; use a clear, well-lit handshape.
