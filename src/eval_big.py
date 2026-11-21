"""사전학습 SigLIP 계열 제로샷 평가: 4,000쌍 전체를 후보 풀로 사용 (학습 없음, 누수 없음)."""
import json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoProcessor

WORK = Path("/kaggle/temp") if Path("/kaggle").exists() else Path("/content/work")
WORK.mkdir(parents=True, exist_ok=True)
DATA = WORK / "haikuthon-train"
snapshot_download(repo_id="KimWooJin/26-haikuthon-train", repo_type="dataset",
                  allow_patterns=["prompts.jsonl", "images/*.png"], local_dir=DATA)
rows = [json.loads(l) for l in (DATA / "prompts.jsonl").open(encoding="utf-8") if l.strip()]
rows.sort(key=lambda r: int(r["id"]))
print("rows", len(rows), flush=True)

MODELS = sys.argv[1:] or [
    "google/siglip-base-patch16-224",
    "google/siglip-large-patch16-256",
    "google/siglip-so400m-patch14-384",
    "google/siglip2-base-patch16-224",
    "google/siglip2-large-patch16-256",
]
dev = "cuda"


def point(rank):
    return 0.0 if rank > 5 else 0.5 * ((rank == 1) + 1 / math.log2(rank + 1))


def load_img(r):
    with Image.open(DATA / "images" / (r["id"] + ".png")) as im:
        return im.convert("RGB")


results = []
for name in MODELS:
    try:
        t0 = time.time()
        net = AutoModel.from_pretrained(name, torch_dtype=torch.float16).to(dev).eval()
        proc = AutoProcessor.from_pretrained(name)
        iv, dec, fwd = [], 0.0, 0.0
        with ThreadPoolExecutor(4) as ex, torch.inference_mode():
            for s in range(0, len(rows), 32):
                t1 = time.time()
                ims = list(ex.map(load_img, rows[s:s + 32]))
                px = proc(images=ims, return_tensors="pt").pixel_values.to(dev, torch.float16)
                dec += time.time() - t1
                t1 = time.time()
                f = net.get_image_features(pixel_values=px)
                f = getattr(f, "pooler_output", f)
                iv.append(torch.nn.functional.normalize(f.float(), dim=-1))
                torch.cuda.synchronize()
                fwd += time.time() - t1
            iv = torch.cat(iv)
            tv = []
            for s in range(0, len(rows), 128):
                tok = proc(text=[r["prompts"].lower() if "siglip2" in name else r["prompts"] for r in rows[s:s + 128]],
                           padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
                f = net.get_text_features(**tok)
                f = getattr(f, "pooler_output", f)
                tv.append(torch.nn.functional.normalize(f.float(), dim=-1))
            S = (torch.cat(tv) @ iv.T).cpu().numpy()
        ranks = 1 + (S > S.diagonal()[:, None]).sum(1)
        res = {"model": name, "score_pool4000": float(np.mean([point(int(r)) for r in ranks])),
               "recall1": float((ranks == 1).mean()), "sec_decode_prep": round(dec, 1),
               "sec_forward": round(fwd, 1), "sec_total": round(time.time() - t0, 1)}
        # 풀 800 (앞 800장)도 같이: 기준선 검증과 비교용
        S8 = S[:800, :800]
        r8 = 1 + (S8 > S8.diagonal()[:, None]).sum(1)
        res["score_pool800"] = float(np.mean([point(int(r)) for r in r8]))
        print(json.dumps(res, ensure_ascii=False), flush=True)
        results.append(res)
        del net
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        print("FAIL", name, repr(e)[:300], flush=True)
Path("/kaggle/working/eval_big.json" if Path("/kaggle/working").exists() else WORK / "eval_big.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=1))
