"""Shrink a training checkpoint to inference-only weights (~275 MB -> ~95 MB).
Training checkpoints also store optimizer/scheduler state, which the app never uses.

Usage (Colab or local):
  python slim_checkpoint.py best_cbam_mtl_model.pth best_cbam_mtl_model_slim.pth [--half]
--half stores fp16 weights (~48 MB); accuracy is normally unchanged, but check /health and a few images.
"""
import sys, torch

src, dst = sys.argv[1], sys.argv[2]
half = "--half" in sys.argv
ck = torch.load(src, map_location="cpu", weights_only=False)
sd = ck["model_state_dict"]
if half:
    sd = {k: v.half() if v.is_floating_point() else v for k, v in sd.items()}
keep = {"model_state_dict": sd}
for k in ("epoch", "current_val_loss", "fruit_classes", "quality_classes"):
    if k in ck:
        keep[k] = ck[k]
torch.save(keep, dst)
print("kept keys:", list(keep), "| dropped:", [k for k in ck if k not in keep])
