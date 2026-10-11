"""FruitAI backend: ResNet50+CBAM multi-task model (+ optional Qwen2.5-VL explanation)."""
import io, os, json, re
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms as T
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.getenv("CKPT_PATH", os.path.join(HERE, "best_cbam_mtl_model.pth"))  # looks next to app.py
USE_QWEN = os.getenv("USE_QWEN", "0") == "1"       # needs a GPU
FRUITS = ["Apple", "Banana", "Guava", "Lime", "Orange", "Pomegranate"]
QUALITIES = ["Bad", "Good", "Mixed"]                  # same order as training
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ChannelAttention(nn.Module):
    def __init__(s, c=2048, r=16):
        super().__init__(); h = max(c // r, 1)
        s.avg_pool = nn.AdaptiveAvgPool2d(1); s.max_pool = nn.AdaptiveMaxPool2d(1)
        s.mlp = nn.Sequential(nn.Conv2d(c, h, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(h, c, 1, bias=False))
        s.sigmoid = nn.Sigmoid()
    def forward(s, x):
        return x * s.sigmoid(s.mlp(s.avg_pool(x)) + s.mlp(s.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(s, k=7):
        super().__init__(); s.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False); s.sigmoid = nn.Sigmoid()
    def forward(s, x):
        a = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1)
        return x * s.sigmoid(s.conv(a))

class CBAM(nn.Module):
    def __init__(s, c=2048):
        super().__init__(); s.channel = ChannelAttention(c); s.spatial = SpatialAttention()
    def forward(s, x): return s.spatial(s.channel(x))

class ResNet50_CBAM_MTL(nn.Module):
    def __init__(s, trained_resnet, num_fruits=6, num_qualities=3):
        super().__init__()
        s.backbone = nn.Sequential(*list(trained_resnet.children())[:-2])
        s.cbam = CBAM(2048); s.pool = nn.AdaptiveAvgPool2d(1)
        s.fruit_head = nn.Linear(2048, num_fruits); s.quality_head = nn.Linear(2048, num_qualities)
    def forward(s, x):
        f = s.pool(s.cbam(s.backbone(x))).flatten(1)
        return s.fruit_head(f), s.quality_head(f)

def ensure_checkpoint():
    """Use local file if present; otherwise download from HF Hub or a direct URL."""
    if os.path.exists(CKPT):
        return
    hf_repo, url = os.getenv("HF_REPO"), os.getenv("CKPT_URL")
    if hf_repo:   # e.g. HF_REPO=yourname/fruitai-weights  (file name = basename of CKPT_PATH)
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(hf_repo, os.path.basename(CKPT), token=os.getenv("HF_TOKEN"))
        os.makedirs(os.path.dirname(CKPT) or ".", exist_ok=True)
        if os.path.abspath(p) != os.path.abspath(CKPT):
            import shutil; shutil.copyfile(p, CKPT)
    elif url:     # direct download link (Drive links need the "uc?export=download&id=..." form)
        import urllib.request
        urllib.request.urlretrieve(url, CKPT)
    else:
        raise FileNotFoundError(
            f"Checkpoint not found at '{CKPT}'. Put best_cbam_mtl_model.pth next to app.py, "
            "or set HF_REPO or CKPT_URL.")

ensure_checkpoint()
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
# The notebook saved fruit/quality class lists in the checkpoint; use them if present.
FRUITS = ck.get("fruit_classes", FRUITS); QUALITIES = ck.get("quality_classes", QUALITIES)
base = models.resnet50(weights=None); base.fc = nn.Linear(2048, 6)
model = ResNet50_CBAM_MTL(base, len(FRUITS), len(QUALITIES))
model.load_state_dict(ck["model_state_dict"]); model.to(dev).eval()
CKPT_INFO = {"epoch": ck.get("epoch"), "val_loss": ck.get("current_val_loss"), "classes": [FRUITS, QUALITIES]}

tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

qwen = None
def load_qwen():
    global qwen
    if qwen is None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        name = "Qwen/Qwen2.5-VL-3B-Instruct"
        qwen = (Qwen2_5_VLForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16, device_map="auto"),
                AutoProcessor.from_pretrained(name))
    return qwen

def explain(img, fruit, fc, quality, qc):
    prompt = f"""You are explaining a trained fruit model's output. Prediction: {fruit} ({fc:.1%}), quality {quality} ({qc:.1%}).
Do not change the labels. Describe only visible features. Do not claim appearance proves food safety. Do not invent measurements.
Return valid JSON only with keys: "visual_evidence", "limitations", "summary"."""
    from qwen_vl_utils import process_vision_info
    m, p = load_qwen()
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    text = p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(msgs)
    inp = p(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to(m.device)
    with torch.inference_mode():
        out = m.generate(**inp, max_new_tokens=300, do_sample=False)
    raw = p.batch_decode([o[len(i):] for i, o in zip(inp.input_ids, out)], skip_special_tokens=True)[0]
    return json.loads(re.search(r"\{.*\}", raw, re.S).group(0))

COLORS = {"Red": (200, 30, 40), "Yellow": (230, 200, 40), "Green": (90, 160, 50), "Orange": (240, 140, 30),
          "Brown": (120, 80, 40), "Dark": (40, 30, 30), "Pink": (220, 120, 140)}
def dominant_color(img):
    c = img.resize((32, 32)).crop((8, 8, 24, 24)).resize((1, 1)).getpixel((0, 0))
    return min(COLORS, key=lambda k: sum((a - b) ** 2 for a, b in zip(COLORS[k], c)))

ATTR = {"Good": ("Smooth", "None", "Fresh"), "Mixed": ("Uneven", "Minor", "Mixed"), "Bad": ("Rough", "Visible", "Deteriorated")}

app = FastAPI(title="FruitAI API")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","), allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health(): return {"ok": True, "device": str(dev), "qwen": USE_QWEN, "checkpoint": CKPT_INFO}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try: img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception: raise HTTPException(400, "Could not read image. Use JPG, PNG or JPEG.")
    with torch.inference_mode():
        fl, ql = model(tf(img).unsqueeze(0).to(dev))
        fp, qp = F.softmax(fl, 1)[0], F.softmax(ql, 1)[0]
    fi, qi = int(fp.argmax()), int(qp.argmax())
    fruit, quality, fc, qc = FRUITS[fi], QUALITIES[qi], float(fp[fi]), float(qp[qi])
    surface, damage, texture = ATTR[quality]
    ex = None
    if USE_QWEN:
        try: ex = explain(img, fruit, fc, quality, qc)
        except Exception: ex = None
    if ex is None:
        ex = {"visual_evidence": f"The model sees a {fruit.lower()} with {surface.lower()} skin and a mostly {dominant_color(img).lower()} colour.",
              "limitations": "Appearance alone cannot confirm freshness or food safety. Use a clear, well-lit photo.",
              "summary": f"Predicted as a {quality.lower()} quality {fruit.lower()}."}
    return {"fruit": fruit, "fruit_confidence": fc, "quality": quality, "quality_confidence": qc,
            "attributes": {"color": dominant_color(img), "surface": surface, "damage": damage, "texture": texture},
            "explanation": ex}
