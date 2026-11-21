"""파서 관계 문구 리랭킹을 포함한 추론 (제출용 predict 후보 + 시간 측정).

점수 = z(전체 문장) + lam * mean(z(관계 문구)). z는 쿼리(문구)마다 후보 풀 안에서만 구한다.
모든 행렬 연산은 GPU에서 청크 단위로 처리한다.
사용: pred, times = predict_rerank((network, processor), queries, candidates, read_image)
"""
import math
import time

import torch
from torch.nn.functional import normalize

from rerank_eval import build_parts


def _zrow(M):
    return (M - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-6)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def predict_rerank(model, queries, candidates, read_image, top_k=5, lam=0.5, group="rel"):
    network, processor = model
    dev, dt = network.device, network.dtype
    times = {}
    t = time.time()

    def tick(name):
        nonlocal t
        _sync()
        now = time.time()
        times[name] = round(now - t, 1)
        t = now

    def enc(texts, bs=256):
        outs = []
        for s in range(0, len(texts), bs):
            tok = processor(text=texts[s:s + bs], padding="max_length", truncation=True,
                            return_tensors="pt").to(dev)
            outs.append(normalize(network.get_text_features(**tok).pooler_output.float(), dim=-1))
        return torch.cat(outs)

    # 1) 이미지 임베딩 (디코딩 + 인코딩)
    vecs = []
    for s in range(0, len(candidates), 32):
        ims = [read_image(c["image_path"]) for c in candidates[s:s + 32]]
        px = processor(images=ims, return_tensors="pt").pixel_values.to(device=dev, dtype=dt)
        vecs.append(normalize(network.get_image_features(pixel_values=px).pooler_output.float(), dim=-1))
    I = torch.cat(vecs)
    tick("이미지")

    # 2) 전체 문장 텍스트 임베딩
    T = enc([q["prompt"] for q in queries])
    tick("전체문장 텍스트")

    # 3) 파서로 부분 문구 만들기 (CPU)
    parts = [build_parts(q["prompt"])[group] for q in queries]
    uniq = sorted({p for ps in parts for p in ps})
    pid = {p: i for i, p in enumerate(uniq)}
    tick("파서")

    # 4) 부분 문구 임베딩 + 후보 풀 안 z-점수
    U = len(uniq)
    if U:
        PT = enc(uniq)
        PZ = torch.empty(U, I.shape[0], dtype=torch.half, device=dev)
        for s in range(0, U, 1024):
            PZ[s:s + 1024] = _zrow(PT[s:s + 1024] @ I.T).half()
    tick("부분문구 인코딩")

    # 5) 쿼리 청크별 점수 합산 + Top-k
    ids = [c["image_id"] for c in candidates]
    pred = {}
    for s in range(0, len(queries), 256):
        qs = queries[s:s + 256]
        F = _zrow(T[s:s + 256] @ I.T)
        if U:
            W = torch.zeros(len(qs), U, dtype=torch.half)
            for r, ps in enumerate(parts[s:s + 256]):
                if ps:
                    idx = torch.tensor([pid[p] for p in ps])
                    W[r].index_add_(0, idx, torch.full((len(ps),), 1.0 / len(ps), dtype=torch.half))
            F = F + lam * (W.to(dev) @ PZ).float()
        top = F.topk(top_k, dim=1).indices.tolist()
        for q, row in zip(qs, top):
            pred[q["prompt_id"]] = [ids[j] for j in row]
    tick("점수합산+순위")
    times["합계"] = round(sum(times.values()), 1)
    print("추론 시간(초):", times, "| 부분문구", U, "개", flush=True)
    return pred, times


def score_valid(pred, valid_rows):
    """검증 800쌍 점수: (평균, Recall@1)."""
    s, h = [], []
    for r in valid_rows:
        top = pred[r["id"]]
        rank = top.index(r["id"]) + 1 if r["id"] in top else None
        s.append(0.0 if rank is None else 0.5 * ((rank == 1) + 1 / math.log2(rank + 1)))
        h.append(rank == 1)
    return sum(s) / len(s), sum(h) / len(h)
