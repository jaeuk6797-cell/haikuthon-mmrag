"""파서 결합 리랭킹 실험 (검증 800쌍).

사용(Colab/Kaggle): 노트북 3단계(model, valid_rows 준비) 뒤에서
    sys.path.insert(0, "/content"); import rerank_eval; rerank_eval.run(model, valid_rows)

아이디어: 전체 프롬프트 점수(SigLIP 코사인)에, 파서로 쪼갠 부분 문구
(객체+속성, 관계 문장) 점수를 더한다. 이미지 임베딩은 한 번만 계산하고
부분 문구는 텍스트 인코딩만 추가되므로 추론 시간 증가는 작다.
각 쿼리는 자기 문구와 후보 풀만 쓴다(다른 쿼리 예측으로 보정하지 않음).
"""
import math
import re
import numpy as np
import torch
from torch.nn.functional import normalize

from parse_prompt import parse_prompt

NUM = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
PRED = {"left_of": "to the left of", "right_of": "to the right of", "above": "above",
        "below": "below", "next_to": "next to", "on": "resting on", "inside": "inside",
        "in_front_of": "in front of", "behind": "behind", "holds": "holding",
        "wears": "wearing", "rides": "riding"}


def plural(w):
    if re.search(r"(s|x|ch|sh)$", w):
        return w + "es"
    if re.search(r"[^aeiou]y$", w):
        return w[:-1] + "ies"
    return w + "s"


def obj_phrase(o, with_attrs=True):
    attrs = [v for vs in o["attrs"].values() for v in vs] if with_attrs else []
    n = o["count"]
    head = o["label"]
    if n and n > 1:
        return " ".join([NUM.get(n, str(n))] + attrs + [plural(head)])
    return " ".join(["a"] + attrs + [head])


def build_parts(text):
    """-> {"obj": [phrase], "rel": [phrase]}"""
    p = parse_prompt(text)
    objs = p["objects"]
    out = {"obj": [], "rel": []}
    for o in objs:
        if o["implicit"]:
            continue
        out["obj"].append(obj_phrase(o))
    for r in p["relations"]:
        pred = PRED.get(r["pred"])
        if pred is None:
            continue
        out["rel"].append("%s %s %s" % (obj_phrase(objs[r["subj"]]), pred, obj_phrase(objs[r["obj"]])))
    return out


def point_score(rank):
    if rank is None or rank > 5:
        return 0.0
    return 0.5 * ((rank == 1) + 1 / math.log2(rank + 1))


def evaluate(F, idx=None):
    """F[i, j]: 쿼리 i vs 이미지 j, 정답은 j == i."""
    F = np.asarray(F)
    idx = range(F.shape[0]) if idx is None else idx
    s, h = [], []
    for i in idx:
        rank = 1 + int((F[i] > F[i, i]).sum())
        s.append(point_score(rank))
        h.append(rank == 1)
    return float(np.mean(s)), float(np.mean(h))


def zrow(M):
    return (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-6)


@torch.inference_mode()
def run(model, valid_rows, read_image, lambdas=(0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)):
    network, processor = model
    dev = network.device

    def enc_text(texts, bs=256):
        outs = []
        for s in range(0, len(texts), bs):
            tok = processor(text=texts[s:s + bs], padding="max_length", truncation=True,
                            return_tensors="pt").to(dev)
            f = network.get_text_features(**tok).pooler_output
            outs.append(normalize(f.float(), dim=-1).cpu())
        return torch.cat(outs).numpy()

    # 이미지 임베딩 (한 번만)
    vecs = []
    for s in range(0, len(valid_rows), 32):
        ims = [read_image(r["image_path"]) for r in valid_rows[s:s + 32]]
        px = processor(images=ims, return_tensors="pt").pixel_values.to(device=dev, dtype=network.dtype)
        f = network.get_image_features(pixel_values=px).pooler_output
        vecs.append(normalize(f.float(), dim=-1).cpu())
    I = torch.cat(vecs).numpy()

    prompts = [r["prompts"] for r in valid_rows]
    S_full = enc_text(prompts) @ I.T
    print("전체 문장만(기준선 재현): 평균 %.4f | Recall@1 %.4f" % evaluate(S_full))

    parts = [build_parts(t) for t in prompts]
    uniq = sorted({p for d in parts for k in ("obj", "rel") for p in d[k]})
    pid = {p: i for i, p in enumerate(uniq)}
    P = zrow(enc_text(uniq) @ I.T)          # 문구마다 후보 풀 안에서 z-점수
    print("부분 문구 %d개 (프롬프트당 객체 %.2f, 관계 %.2f)" % (
        len(uniq), np.mean([len(d["obj"]) for d in parts]), np.mean([len(d["rel"]) for d in parts])))

    Z = zrow(S_full)

    def part_matrix(groups, agg):
        M = np.zeros_like(S_full)
        for i, d in enumerate(parts):
            ids = [pid[p] for g in groups for p in d[g]]
            if not ids:
                continue
            rows = P[ids]
            M[i] = rows.mean(0) if agg == "mean" else rows.min(0)
        return M

    half = len(valid_rows) // 2
    results = []
    for groups in (("obj",), ("rel",), ("obj", "rel")):
        for agg in ("mean", "min"):
            M = part_matrix(groups, agg)
            for lam in lambdas:
                F = Z + lam * M
                results.append((groups, agg, lam, *evaluate(F), evaluate(F, range(half))[0],
                                evaluate(F, range(half, len(valid_rows)))[0]))
    print("%-10s %-5s %5s | %7s %7s | %7s %7s" % ("부분", "집계", "λ", "평균", "R@1", "앞400", "뒤400"))
    for g, a, l, s, h, s1, s2 in results:
        print("%-10s %-5s %5.2f | %7.4f %7.4f | %7.4f %7.4f" % ("+".join(g), a, l, s, h, s1, s2))
    best = max(results, key=lambda x: x[3])
    print("최고:", best)
    return {"S_full": S_full, "P": P, "parts": parts, "results": results}
