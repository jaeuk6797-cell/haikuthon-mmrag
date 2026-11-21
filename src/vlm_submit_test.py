"""model_vlm.py 검증·시간 측정 (Colab). exec 후 prep(), valid_scan(), big_time(thr) 순서로 호출."""
import importlib, json, shutil, sys, time
from pathlib import Path
import numpy as np, torch

SUB = Path("/content/subv")

def prep(ckpt="/content/g_u8"):
    (SUB / "models").mkdir(parents=True, exist_ok=True)
    for f in Path(ckpt).iterdir():
        if f.name not in ("train_log.json", "valid_ids.json"):
            shutil.copy(f, SUB / "models" / f.name)
    shutil.copy("/content/parse_prompt.py", SUB)
    shutil.copy("/content/model_vlm.py", SUB / "model.py")
    if not (SUB / "vlm").exists():
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        n = "Qwen/Qwen3-VL-Reranker-2B"
        AutoProcessor.from_pretrained(n).save_pretrained(SUB / "vlm")
        Qwen3VLForConditionalGeneration.from_pretrained(n, dtype=torch.float16).save_pretrained(SUB / "vlm")
    print("prep ok", [p.name for p in SUB.iterdir()])

def _load():
    sys.path.insert(0, str(SUB))
    for m in ("model", "parse_prompt"):
        sys.modules.pop(m, None)
    import model as M
    return M, M.load_model(str(SUB), "cuda")

def valid_scan(ckpt="/content/g_u8", data=None):
    data = Path(data)
    rows = {}
    for l in open(data / "prompts.jsonl", encoding="utf-8"):
        r = json.loads(l); rows[r["id"]] = r
    ids = json.load(open(Path(ckpt) / "valid_ids.json"))
    q = [{"prompt_id": i, "prompt": rows[i]["prompts"]} for i in ids]
    c = [{"image_id": i, "image_path": str(data / "images" / ("%s.png" % i))} for i in ids]
    M, mdl = _load()
    M.MARGIN_THR = 1e9
    t = time.time(); pred = M.predict(mdl, q, c); print("검증 800 전체 VLM 포함 %.0f초" % (time.time() - t), flush=True)
    pt = lambda r: 0.0 if r > 5 else 0.5 * ((r == 1) + 1 / np.log2(r + 1))
    def score(preds):
        s = []
        for i in ids:
            top = preds[i]; s.append(pt(top.index(i) + 1) if i in top else 0.0)
        return float(np.mean(s))
    # 임계값별: 원 순위 = VLM 없이 (swap 되돌림)
    base = {}
    margins = np.array(M.LAST["margin"]); vlm = M.LAST["vlm"]
    for k, i in enumerate(ids):
        top = pred[i]
        if vlm[k][1] > vlm[k][0]:  # 바뀐 것을 되돌려 원래 1,2위 복원
            top = [top[1], top[0]] + top[2:]
        base[i] = top
    print("VLM 없이(관계 재정렬만) %.4f" % score(base))
    for q_ in (0.1, 0.25, 0.4, 0.5, 0.75, 1.0):
        thr = float(np.quantile(margins, q_)) + 1e-9
        new = {}
        for k, i in enumerate(ids):
            top = list(base[i])
            if margins[k] <= thr and vlm[k][1] > vlm[k][0]:
                top[0], top[1] = top[1], top[0]
            new[i] = top
        print("게이트 %3.0f%% thr %.5f | 점수 %.4f" % (100 * q_, thr, score(new)), flush=True)

def big_time(thr, big_queries, big_candidates):
    M, mdl = _load()
    M.MARGIN_THR = thr
    torch.cuda.synchronize(); t = time.time()
    pred = M.predict(mdl, big_queries, big_candidates)
    torch.cuda.synchronize()
    print("16k·8k 전체 %.0f초 | 게이트 %s | 바뀜 %s" % (time.time() - t, M.LAST.get("gated"), M.LAST.get("swapped")), flush=True)
