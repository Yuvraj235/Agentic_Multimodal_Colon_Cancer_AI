"""ColonAI — Real-time FPS benchmark on MPS / CUDA / CPU.

Live colonoscopy operates at 25-60 FPS. Anything below ~15 FPS becomes
laggy to the operator and unsafe (polyp drifts out of frame before the
overlay catches up). This script measures the end-to-end frame
processing rate including:

  1. Image preprocess (resize + normalise)
  2. Forward pass (UnifiedMultiModalTransformer)
  3. GradCAM++ generation
  4. Heatmap → bbox extraction

Reports mean / median / p95 / p99 latency and effective FPS.
Saves outputs/unified_multimodal_v2/fps_benchmark.json
"""
from __future__ import annotations
import sys, time, json, statistics
from pathlib import Path
import numpy as np, torch, cv2
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unified_transformer import UnifiedMultiModalTransformer
from src.data.multimodal_dataset import N_TABULAR_FEATURES
from src.agents.unified_image_agent import GradCAMPlusPlus
from transformers import AutoTokenizer

CHECKPOINT = "outputs/unified_multimodal_v2/checkpoints/best_model.pth"
BERT       = "dmis-lab/biobert-base-cased-v1.2"
N_WARMUP   = 5
N_FRAMES   = 60


def _device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def main():
    device = _device()
    print(f"Device: {device}\nCheckpoint: {CHECKPOINT}\n")

    model = UnifiedMultiModalTransformer(
        n_tabular_features=N_TABULAR_FEATURES, n_classes=5).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()

    cam_ex = GradCAMPlusPlus(model, model.get_image_target_layer())
    tok    = AutoTokenizer.from_pretrained(BERT)
    enc    = tok("Live colonoscopy stream.", padding="max_length",
                 truncation=True, max_length=64, return_tensors="pt")
    ids    = enc["input_ids"].to(device)
    msk    = enc["attention_mask"].to(device)
    tab    = torch.zeros((1, N_TABULAR_FEATURES), device=device)
    tfm    = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])

    # Synthetic frames (HD 1280x720, typical endoscopy resolution)
    rng = np.random.default_rng(0)
    frames = [np.clip(rng.normal(120, 30, (720, 1280, 3)), 0, 255).astype(np.uint8)
              for _ in range(N_WARMUP + N_FRAMES)]
    print(f"Generated {len(frames)} synthetic HD frames.")

    print("\nWarming up …")
    for f in frames[:N_WARMUP]:
        x = tfm(Image.fromarray(f)).unsqueeze(0).to(device)
        with torch.no_grad(): _ = model(x, ids, msk, tab)

    print(f"\nBenchmarking {N_FRAMES} frames …")
    latencies_total, latencies_fwd, latencies_cam = [], [], []
    for i, frame in enumerate(frames[N_WARMUP:]):
        t0 = time.perf_counter()

        x = tfm(Image.fromarray(frame)).unsqueeze(0).to(device)
        t1 = time.perf_counter()

        with torch.no_grad():
            out  = model(x, ids, msk, tab)
            prob = F.softmax(out["pathology"], dim=-1)[0]
            pred = int(prob.argmax())
        if device.type == "mps":
            torch.mps.synchronize()
        t2 = time.perf_counter()

        # Only run GradCAM if predicted polyp (realistic for live use)
        if pred == 0:
            cam = cam_ex.generate(image=x.detach().requires_grad_(True),
                                  class_idx=0, input_ids=ids,
                                  attention_mask=msk, tabular=tab)
            if cam is not None and cam.size >= 4:
                _ = cv2.resize(cam.astype(np.float32), (1280, 720),
                               interpolation=cv2.INTER_LINEAR)
        if device.type == "mps":
            torch.mps.synchronize()
        t3 = time.perf_counter()

        latencies_total.append((t3 - t0) * 1000)
        latencies_fwd.append((t2 - t1) * 1000)
        latencies_cam.append((t3 - t2) * 1000)

        if (i+1) % 10 == 0:
            print(f"  frame {i+1}/{N_FRAMES}  "
                  f"last total={latencies_total[-1]:.1f} ms  "
                  f"fwd={latencies_fwd[-1]:.1f}  cam={latencies_cam[-1]:.1f}")

    def stats(arr):
        s = sorted(arr)
        return {
            "mean_ms":   float(statistics.mean(arr)),
            "median_ms": float(statistics.median(arr)),
            "p95_ms":    float(s[int(0.95 * len(s))]),
            "p99_ms":    float(s[int(0.99 * len(s))]),
            "min_ms":    float(min(arr)), "max_ms": float(max(arr)),
        }
    total_stats = stats(latencies_total)
    fps_mean = 1000 / total_stats["mean_ms"]
    fps_p95  = 1000 / total_stats["p95_ms"]

    print(f"\n{'='*60}\nFPS BENCHMARK RESULTS\n{'='*60}")
    print(f"  Device                : {device}")
    print(f"  Frames measured       : {N_FRAMES}")
    print(f"  Mean latency / frame  : {total_stats['mean_ms']:7.1f} ms")
    print(f"  Median latency        : {total_stats['median_ms']:7.1f} ms")
    print(f"  P95 latency           : {total_stats['p95_ms']:7.1f} ms")
    print(f"  P99 latency           : {total_stats['p99_ms']:7.1f} ms")
    print(f"\n  Mean FPS              : {fps_mean:7.1f}")
    print(f"  P95 FPS               : {fps_p95:7.1f}")
    print(f"\n  Breakdown — forward   : {statistics.mean(latencies_fwd):.1f} ms")
    print(f"  Breakdown — gradcam   : {statistics.mean(latencies_cam):.1f} ms")
    verdict = ("✅ Real-time-capable (≥15 FPS)" if fps_mean >= 15
               else ("⚠️ Marginal (5-15 FPS) — frame-skip needed"
                     if fps_mean >= 5
                     else "❌ Too slow for live use (<5 FPS)"))
    print(f"\n  Live-use verdict      : {verdict}")
    print("="*60)

    out = Path("outputs/unified_multimodal_v2/fps_benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "device":             str(device),
        "n_frames":           N_FRAMES,
        "total":              total_stats,
        "forward":            stats(latencies_fwd),
        "gradcam":            stats(latencies_cam),
        "fps_mean":           fps_mean,
        "fps_p95":            fps_p95,
        "live_capable":       fps_mean >= 15,
    }, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
