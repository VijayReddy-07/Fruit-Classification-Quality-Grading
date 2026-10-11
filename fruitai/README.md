# FruitAI – deploy guide

Vercel can't run PyTorch/Qwen (size + no GPU), so deploy in two parts.

## 1. Backend (your model) – Hugging Face Spaces (Docker) or Render
1. Get `best_cbam_mtl_model.pth` from `MyDrive/CBAM_MTL_Checkpoints/` (275.7 MB). **Shrink it first** with `python backend/slim_checkpoint.py best_cbam_mtl_model.pth best_cbam_mtl_model.pth.slim` then rename the result to `best_cbam_mtl_model.pth` (about 95 MB; training state removed). You can run this in Colab right after downloading from Drive. Pick one way to supply it:
   - **Local file:** put it in `backend/` (files >100MB need Git LFS).
   - **Hugging Face Hub (recommended):** upload it to a model repo, then set `HF_REPO=yourname/fruitai-weights` (add `HF_TOKEN` if private).
   - **Direct link:** set `CKPT_URL=https://...`. The backend downloads it on first start.
   Optional: `CKPT_PATH` to change the file name or location. Check `/health` to confirm the checkpoint loaded.
2. Run locally to test:
   ```
   cd backend && pip install -r requirements.txt
   uvicorn app:app --port 8000
   ```
3. Deploy `backend/` using the included `Dockerfile` (Hugging Face Spaces → New Space → Docker; port 7860). For Render, use the same Dockerfile.
4. Env vars: `CORS_ORIGINS=https://your-app.vercel.app`, optional `USE_QWEN=1` (needs a GPU; uncomment the extra packages in requirements.txt).
   Without Qwen, the explanation is a short template based on the model's output.

## 2. Frontend – Vercel
1. In `frontend/index.html`, set `API_URL` to your backend URL.
2. Optional: add `frontend/examples/apple.jpg`, `banana.jpg`, `guava.jpg`, `lime.jpg`, `orange.jpg`, `pomegranate.jpg` for the example buttons.
3. `cd frontend && npx vercel --prod` (or import the repo in Vercel and set **Root Directory** to `frontend`). No build step.
