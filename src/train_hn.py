"""어려운 오답 배치 학습 (스타터킷 train.py 대체).

스타터킷 대비 바뀐 점
1) 같은 객체 집합(파서 기준)의 프롬프트끼리 한 배치에 묶어서, 속성/관계만 다른 어려운 오답이 배치 안에 들어가게 함
2) 출력 헤드 + 마지막 N개 트랜스포머 층까지 학습 (--unfreeze)
3) 이미지를 한 번만 전처리해 메모리에 캐시 (에포크마다 PNG 디코딩 안 함)
4) fp16 autocast + GradScaler, 좌우 반전 같은 증강은 쓰지 않음 (좌/우 관계가 뒤집힘)
사용: python train_hn.py --kit /kaggle/temp/starterkit --data /kaggle/temp/haikuthon-train
"""
import argparse
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, SiglipModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_prompt import parse_prompt  # noqa: E402


def point(rank):
    return 0.0 if rank > 5 else 0.5 * ((rank == 1) + 1 / math.log2(rank + 1))


def object_key(text):
    p = parse_prompt(text)
    return tuple(sorted(o["label"] for o in p["objects"]))


def cache_pixels(rows, processor, path):
    if Path(path).exists():
        return torch.load(path)
    from PIL import Image

    def load(r):
        with Image.open(r["image_path"]) as im:
            return processor(images=im.convert("RGB"), return_tensors="pt").pixel_values[0].half()

    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        pix = torch.stack(list(ex.map(load, rows)))
    print("이미지 전처리 %.0f초, %s" % (time.time() - t0, tuple(pix.shape)), flush=True)
    torch.save(pix, path)
    return pix


def tok(processor, texts, dev):
    return processor(text=texts, padding="max_length", truncation=True, return_tensors="pt").to(dev)


# 규칙 기반 hard-negative 문장 (팀원 group22의 train_siglip2_kaggle.py 방식을 옮김):
# 같은 종류 속성 두 개를 맞바꾸거나 관계 방향을 뒤집어, 같은 이미지에 어긋나는 문장을 만든다.
GROUPS = [
    "red orange yellow green blue purple pink brown black white gray grey beige silver gold".split(),
    "striped polka-dotted solid-colored plaid checkered floral".split(),
    "metal plastic wooden wood glass ceramic leather fabric paper stone".split(),
    "round square oval rectangular triangular".split(),
    "open closed".split(),
    "standing sitting lying".split(),
]
FLIPS = [("left", "right"), ("above", "below")]


def _swap(p, a, b):
    return re.sub(r"\b(%s|%s)\b" % (re.escape(a), re.escape(b)), lambda m: b if m.group(0) == a else a, p)


def perturb(p, rng):
    opts = []
    for g in GROUPS:
        present = [w for w in g if re.search(r"\b%s\b" % re.escape(w), p)]
        if len(present) >= 2:
            opts.append(tuple(rng.sample(present, 2)))
    opts += [f for f in FLIPS if re.search(r"\b(%s|%s)\b" % f, p)]
    if not opts:
        return None
    q = _swap(p, *rng.choice(opts))
    return q if q != p else None


def make_batches(rows, keys, bs, rng):
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    order = list(groups.values())
    rng.shuffle(order)
    flat = []
    for g in order:
        rng.shuffle(g)
        flat.extend(g)
    return [flat[s:s + bs] for s in range(0, len(flat), bs)]


@torch.inference_mode()
def validate(net, processor, pix, rows, dev):
    net.eval()
    iv = []
    for s in range(0, len(rows), 64):
        with torch.autocast("cuda", dtype=torch.float16):
            f = net.get_image_features(pixel_values=pix[s:s + 64].to(dev)).pooler_output
        iv.append(torch.nn.functional.normalize(f.float(), dim=-1))
    iv = torch.cat(iv)
    tv = []
    for s in range(0, len(rows), 128):
        t = tok(processor, [r["prompts"] for r in rows[s:s + 128]], dev)
        with torch.autocast("cuda", dtype=torch.float16):
            f = net.get_text_features(**t).pooler_output
        tv.append(torch.nn.functional.normalize(f.float(), dim=-1))
    S = (torch.cat(tv) @ iv.T).cpu().numpy()
    ranks = 1 + (S > S.diagonal()[:, None]).sum(1)
    return float(np.mean([point(int(r)) for r in ranks])), float((ranks == 1).mean()), S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", type=Path, default=None, help="없으면 스타터킷 없이 HF 모델 ID와 prompts.jsonl을 직접 사용")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("ckpt_hn"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--unfreeze", type=int, default=4)
    ap.add_argument("--hard", type=int, default=1, help="1: 같은 객체 집합끼리 묶음, 0: 무작위 배치")
    ap.add_argument("--all", type=int, default=0, help="1: 4000쌍 전부로 학습(검증 없음, 마지막 에포크 저장)")
    ap.add_argument("--perturb", type=int, default=0, help="1: 속성 맞바꾼 문장을 음성으로 추가")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="random", choices=["random", "group"], help="group: 객체 조합 단위 분할")
    a = ap.parse_args()

    if a.kit:
        sys.path.insert(0, str(a.kit))
        from train import MODEL_ID, load_and_split
        train, valid = load_and_split(a.data, 42)
    else:
        MODEL_ID = "google/siglip-base-patch16-224"
        rows = [json.loads(l) for l in open(a.data / "prompts.jsonl", encoding="utf-8")]
        for r in rows:
            r["image_path"] = str(a.data / "images" / ("%s.png" % r["id"]))
        train, valid = rows[:3200], rows[3200:]
    dev = "cuda"
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)

    if a.kit:
        processor = AutoProcessor.from_pretrained(a.kit / "sample_submission" / "models", local_files_only=True)
    else:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
    pix_all = cache_pixels(train + valid, processor, str(a.data.parent / "pix_cache.pt"))
    if a.split == "group":
        # 객체 조합(라벨 집합) 단위로 분할: 검증 쿼리의 조합이 학습에 하나도 없게 한다.
        allr = train + valid
        gk = [object_key(r["prompts"]) for r in allr]
        by = defaultdict(list)
        for i, k in enumerate(gk):
            by[k].append(i)
        gl = sorted(by)
        random.Random(42).shuffle(gl)
        va_idx = []
        for k in gl:
            if len(va_idx) >= 800:
                break
            va_idx += by[k]
        vs = set(va_idx)
        tr_idx = [i for i in range(len(allr)) if i not in vs]
        train, valid = [allr[i] for i in tr_idx], [allr[i] for i in va_idx]
        pix_tr, pix_va = pix_all[tr_idx], pix_all[va_idx]
        tk = {object_key(r["prompts"]) for r in train}
        ov = sum(object_key(r["prompts"]) in tk for r in valid)
        print("그룹 분할: 학습 %d / 검증 %d, 검증 중 학습과 조합이 겹치는 수 %d" % (len(train), len(valid), ov), flush=True)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "valid_ids.json").write_text(json.dumps([r["id"] for r in valid]))
    elif a.all:
        train, valid = train + valid, (train + valid)[:64]
        pix_tr, pix_va = pix_all, pix_all[:64]
        print("전체 학습: %d쌍 (검증 점수는 학습 데이터라 의미 없음)" % len(train), flush=True)
    else:
        pix_tr, pix_va = pix_all[:len(train)], pix_all[len(train):]

    keys = [object_key(r["prompts"]) for r in train]
    sizes = defaultdict(int)
    for k in keys:
        sizes[k] += 1
    print("학습 객체집합 %d개, 집합당 프롬프트 평균 %.2f" % (len(sizes), len(train) / len(sizes)), flush=True)

    net = SiglipModel.from_pretrained(MODEL_ID).to(dev)
    net.requires_grad_(False)
    n = a.unfreeze
    train_mods = [net.vision_model.head, net.text_model.head, net.vision_model.post_layernorm,
                  net.text_model.final_layer_norm]
    if n > 0:
        train_mods += list(net.vision_model.encoder.layers[-n:]) + list(net.text_model.encoder.layers[-n:])
    for m in train_mods:
        m.requires_grad_(True)
    net.logit_scale.requires_grad_(True)
    net.logit_bias.requires_grad_(True)
    params = [p for p in net.parameters() if p.requires_grad]
    print("학습 파라미터 %.1fM" % (sum(p.numel() for p in params) / 1e6), flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    scaler = torch.amp.GradScaler("cuda")

    best, log = -1.0, []
    s0, r0, _ = validate(net, processor, pix_va, valid, dev)
    print("에포크 0 (제로샷) | 평균 %.4f | Recall@1 %.4f" % (s0, r0), flush=True)
    for ep in range(1, a.epochs + 1):
        net.train()
        if a.hard:
            batches = make_batches(train, keys, a.bs, rng)
        else:
            idx = list(range(len(train)))
            rng.shuffle(idx)
            batches = [idx[s:s + a.bs] for s in range(0, len(idx), a.bs)]
        t0, tot = time.time(), 0.0
        for b in batches:
            if len(b) < 4:
                continue
            texts = [train[i]["prompts"] for i in b]
            if a.perturb:
                texts += [q for q in (perturb(x, rng) for x in texts) if q]
            t = tok(processor, texts, dev)
            px = pix_tr[b].to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                if a.perturb:
                    iv = torch.nn.functional.normalize(net.get_image_features(pixel_values=px).pooler_output.float(), dim=-1)
                    tv = torch.nn.functional.normalize(net.get_text_features(**t).pooler_output.float(), dim=-1)
                    lg = tv @ iv.T * net.logit_scale.exp() + net.logit_bias
                    lab = -torch.ones_like(lg)
                    lab[:len(b)] += 2 * torch.eye(len(b), device=dev)
                    loss = -torch.nn.functional.logsigmoid(lab * lg).sum() / len(b)
                else:
                    loss = net(**t, pixel_values=px, return_loss=True).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(b)
        score, r1, _ = validate(net, processor, pix_va, valid, dev)
        log.append({"epoch": ep, "loss": tot / len(train), "score": score, "recall1": r1, "sec": time.time() - t0})
        print("에포크 %d | loss %.4f | 평균 %.4f | Recall@1 %.4f | %.0f초" % (
            ep, tot / len(train), score, r1, time.time() - t0), flush=True)
        if score > best or a.all:
            best = score
            a.out.mkdir(parents=True, exist_ok=True)
            net.save_pretrained(a.out)
            processor.save_pretrained(a.out)
    (a.out / "train_log.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(a).items()}, "log": log}, indent=1))
    print("최고 평균 %.4f, 저장: %s" % (best, a.out), flush=True)


if __name__ == "__main__":
    main()
