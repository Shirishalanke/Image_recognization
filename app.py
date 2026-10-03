"""
Image Recognition App (animated / 3D edition)
Classifies an image into 1,000 ImageNet categories using a pretrained
model from Hugging Face. Runs offline after the first model download.
"""
import base64
import html
import io
import time

import streamlit as st
import torch
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Image Recognition", page_icon="🖼️", layout="centered")

MODELS = {
    "CLIP Base - fast (recommended)": ("openai/clip-vit-base-patch32", "clip"),
    "CLIP Large - most accurate, slow on CPU": ("openai/clip-vit-large-patch14", "clip"),
    "ViT Base (ImageNet objects only)": ("google/vit-base-patch16-224", "imagenet"),
    "ResNet-50 (ImageNet objects only)": ("microsoft/resnet-50", "imagenet"),
}

DEFAULT_LABELS = (
    "a doctor, a nurse, a man, a woman, a child, a baby, a student, a teacher, a chef, "
    "a police officer, a soldier, a firefighter, an athlete, a musician, a farmer, a crowd of people, "
    "a dog, a cat, a bird, a horse, a cow, an elephant, a fish, a butterfly, a flower, a tree, "
    "a laptop, a mobile phone, a camera, a car, a bicycle, a motorbike, a bus, an airplane, a boat, "
    "a book, a chair, a table, a cup, a bottle, a bag, a shoe, a watch, a stethoscope, a lab coat, "
    "pizza, a burger, fruit, a cake, a beach, a mountain, a forest, a city street, a building, an indoor room"
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Sora', sans-serif; }

/* Sunset-plum animated background */
.stApp {
  background: radial-gradient(circle at 15% 10%, #4a1b6d 0, transparent 45%),
              radial-gradient(circle at 85% 90%, #7a2250 0, transparent 45%),
              linear-gradient(160deg, #150a26, #1d0f3a 50%, #2a1033);
  background-attachment: fixed; color: #fdf1ff;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: rgba(21,10,38,.8); backdrop-filter: blur(14px); border-right:1px solid rgba(255,79,163,.25); }
h1, h2, h3, p, label, span, .stMarkdown { color: #fdf1ff; }

.block-container { position:relative; z-index:1; }

/* Hero: 3D cube with people */
.hero { display:flex; align-items:center; gap:30px; padding:6px 0 20px; }
.scene { width:96px; height:96px; perspective:650px; flex:none; margin:12px 26px 12px 12px; }
.cube { width:100%; height:100%; position:relative; transform-style:preserve-3d; animation:spin 28s linear infinite; }
.face {
  position:absolute; width:96px; height:96px; display:flex; align-items:center; justify-content:center; font-size:40px;
  border:1px solid rgba(255,179,71,.8); border-radius:14px;
  background:linear-gradient(135deg, rgba(255,79,163,.28), rgba(255,179,71,.18));
  box-shadow:inset 0 0 24px rgba(255,79,163,.4);
}
.f1{transform:rotateY(0deg)   translateZ(48px)}
.f2{transform:rotateY(90deg)  translateZ(48px)}
.f3{transform:rotateY(180deg) translateZ(48px)}
.f4{transform:rotateY(-90deg) translateZ(48px)}
.f5{transform:rotateX(90deg)  translateZ(48px)}
.f6{transform:rotateX(-90deg) translateZ(48px)}
@keyframes spin { from{transform:rotateX(-22deg) rotateY(0)} to{transform:rotateX(-22deg) rotateY(360deg)} }
.hero h1 {
  margin:0; font-size:2.7rem; font-weight:800; letter-spacing:-.03em;
  background:linear-gradient(90deg, #ff4fa3, #ffb347);
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.hero p { margin:6px 0 0; opacity:.8; }

/* Tabs + uploader */
[data-baseweb="tab-highlight"] { background:#ff4fa3 !important; height:3px; }
[data-testid="stFileUploaderDropzone"] {
  background:rgba(255,255,255,.06); border:2px dashed rgba(255,179,71,.6); border-radius:18px;
  backdrop-filter:blur(8px); transition:transform .3s, box-shadow .3s, border-color .3s;
}
[data-testid="stFileUploaderDropzone"]:hover { transform:translateY(-3px) scale(1.01); border-color:#ff4fa3; box-shadow:0 18px 40px -12px rgba(255,79,163,.55); }
[data-testid="stAlert"] { background:rgba(61,224,255,.1); border:1px solid rgba(61,224,255,.35); border-radius:16px; }

/* Tilt image card */
.tilt-wrap { perspective:900px; margin:10px 0 18px; }
.tilt {
  position:relative; overflow:hidden; border-radius:22px; transform:rotateX(4deg) rotateY(-6deg); transform-style:preserve-3d;
  transition:transform .5s cubic-bezier(.2,.8,.2,1), box-shadow .5s;
  box-shadow:0 30px 60px -20px rgba(0,0,0,.75), 0 0 0 1px rgba(255,255,255,.1);
}
.tilt:hover { transform:rotateX(0) rotateY(0) scale(1.02); box-shadow:0 40px 80px -20px rgba(255,79,163,.6), 0 0 0 2px rgba(255,179,71,.6); }
.tilt img { display:block; width:100%; }
.scan::after { content:""; position:absolute; left:0; right:0; height:4px; top:0; background:#3de0ff; box-shadow:0 0 20px 8px rgba(61,224,255,.75); animation:sweep 1.5s ease-in-out infinite alternate; }
.scan::before { content:""; position:absolute; inset:0; background:rgba(21,10,38,.35); z-index:1; }
@keyframes sweep { from{top:0} to{top:calc(100% - 4px)} }

/* Results */
.top {
  padding:20px 24px; border-radius:20px; margin-bottom:14px;
  background:linear-gradient(135deg, rgba(255,79,163,.25), rgba(255,179,71,.2));
  border:1px solid rgba(255,179,71,.6);
}
.top small { opacity:.75; }
.top b { font-size:1.7rem; display:block; margin-top:2px; }
.top .pct { color:#ffb347; font-weight:800; }
.row { margin:12px 0; }
.row .lab { display:flex; justify-content:space-between; font-size:.95rem; margin-bottom:5px; }
.track { height:11px; border-radius:99px; background:rgba(255,255,255,.1); overflow:hidden; }
.fill { height:100%; width:0; border-radius:99px; background:linear-gradient(90deg, #ff4fa3, #ffb347); box-shadow:0 0 14px rgba(255,79,163,.7); animation:grow 1.1s cubic-bezier(.2,.8,.2,1) forwards; }
@keyframes grow { to{ width:var(--w) } }

/* 3D button */
.stButton > button {
  border:0; border-radius:16px; padding:.75rem 1.6rem; font-weight:700; color:#fff;
  background:linear-gradient(135deg, #ff4fa3, #ff8a4c);
  box-shadow:0 8px 0 #9c1f62, 0 16px 26px rgba(0,0,0,.45); transition:transform .15s, box-shadow .15s;
}
.stButton > button:hover { transform:translateY(-2px); color:#fff; }
.stButton > button:active { transform:translateY(6px); box-shadow:0 2px 0 #9c1f62, 0 4px 10px rgba(0,0,0,.4); }

@media (prefers-reduced-motion: reduce) {
  .cube, .scan::after, .fill { animation:none !important; }
  .fill { width:var(--w); }
}
</style>
"""

HERO = """
<div class="hero">
<div class="scene"><div class="cube">
<div class="face f1">🧑‍🚀</div><div class="face f2">👩‍🎤</div><div class="face f3">🧑‍🍳</div>
<div class="face f4">👨‍🔬</div><div class="face f5">🧑‍🎨</div><div class="face f6">👩‍💻</div>
</div></div>
<div><h1>Image Recognition</h1><p>Show me something 👋 I'll tell you what I see. Upload a photo or snap one with your camera.</p></div>
</div>
"""


@st.cache_resource(show_spinner="Loading model (first run downloads it, please wait)...")
def load_classifier(model_id: str, kind: str):
    task = "zero-shot-image-classification" if kind == "clip" else "image-classification"
    if torch.cuda.is_available():
        return pipeline(task, model=model_id, device=0, torch_dtype=torch.float16)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return pipeline(task, model=model_id)


def to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def image_card(img: Image.Image, scanning: bool = False) -> str:
    cls = "tilt scan" if scanning else "tilt"
    return f'<div class="tilt-wrap"><div class="{cls}"><img src="{to_data_uri(img)}"></div></div>'


def results_html(results) -> str:
    best = results[0]
    out = (
        f'<div class="top"><small>Top prediction</small>'
        f'<b>{html.escape(best["label"])} <span class="pct">{best["score"] * 100:.1f}%</span></b></div>'
    )
    for i, r in enumerate(results):
        pct = r["score"] * 100
        d = 0
        out += (
            f'<div class="row" style="animation-delay:{d}s">'
            f'<div class="lab"><span>{html.escape(r["label"])}</span><span>{pct:.2f}%</span></div>'
            f'<div class="track"><div class="fill" style="--w:{pct:.2f}%; animation-delay:{d + 0.2}s"></div></div>'
            f"</div>"
        )
    return out


st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
top_k = st.sidebar.slider("Number of predictions", 1, 10, 5)
model_id, kind = MODELS[model_name]
labels_text = DEFAULT_LABELS
if kind == "clip":
    labels_text = st.sidebar.text_area(
        "What can it look for? (comma separated)", DEFAULT_LABELS, height=220,
        help="CLIP can recognize anything you describe. Add specific words for better results.",
    )
st.sidebar.markdown("---")
st.sidebar.caption("CLIP models match your photo against the words above. ImageNet models only know 1,000 object classes and have no \"person\" class.")

# ---------------- Main ----------------
tab_upload, tab_camera = st.tabs(["🙋 Upload", "🤳 Camera"])
image = None

with tab_upload:
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if file:
        image = Image.open(file).convert("RGB")

with tab_camera:
    photo = st.camera_input("Take a picture")
    if photo:
        image = Image.open(photo).convert("RGB")

if image is not None:
    card = st.empty()
    card.markdown(image_card(image), unsafe_allow_html=True)

    if st.button("🧑‍💻 Recognize", type="primary"):
        classifier = load_classifier(model_id, kind)  # instant once cached
        card.markdown(image_card(image, scanning=True), unsafe_allow_html=True)
        small = image.copy()
        small.thumbnail((512, 512))  # models resize to 224px anyway; this just skips wasted work
        t0 = time.time()
        with torch.inference_mode():
            if kind == "clip":
                labels = [x.strip() for x in labels_text.split(",") if x.strip()]
                raw = classifier(small, candidate_labels=labels, hypothesis_template="a photo of {}.")
                results = sorted(raw, key=lambda r: r["score"], reverse=True)[:top_k]
            else:
                results = classifier(small, top_k=top_k)
        elapsed = time.time() - t0
        card.markdown(image_card(image), unsafe_allow_html=True)
        st.markdown(results_html(results), unsafe_allow_html=True)
        st.caption(f"Recognized in {elapsed:.1f}s")
else:
    st.info("🙌 Upload an image or use the camera to get started.")

# Warm up the model AFTER the page is drawn, so the UI never freezes while it loads
load_classifier(model_id, kind)