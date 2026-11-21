"""HAIKUTHON 제출 모델: SigLIP 검색 → 조건 분해 재정렬 → (선택) BLIP ITM 재정렬 → (선택) OWL-ViT 상자로 물체별 색·관계 점검.

1. 후보 이미지를 한 번 읽어 SigLIP으로 인코딩한다 (DataLoader worker로 디코딩 병렬화).
   BLIP을 쓰면 같은 크기로 줄인 이미지를 GPU에 보관해 BLIP도 다시 읽지 않고 쓴다.
2. 질의마다 원문으로 상위 K개 후보를 찾는다.
3. 질의 문장 하나에서 규칙으로 객체별 조건과 대조 문구를 만들고, 후보 K개의 조건 점수를 원래 점수에 더한다.
4. (models/blip이 있고 설정에서 켰을 때) 조건 재정렬 결과의 상위 k개를 BLIP ITM(교차 어텐션)으로 다시 채점해 더한다.
5. (models/owl이 있고 설정에서 켰을 때) 질의에 단일 물체 사이의 좌우·상하 관계가 있으면, 최종 상위 2장에서
   두 물체를 OWL-ViT로 검출해 상자 중심을 비교하고, 2위가 관계를 더 잘 지키면 두 장의 순서만 바꾼다.
   검출은 (후보 이미지, 물체 이름)마다 독립이라 같은 이미지를 보는 질의끼리 계산만 공유한다.
6. (설정 color.use) 같은 OWL 검출 상자를 잘라 SigLIP으로 물체별 색을 비교해 상위 2장의 순서를 정한다.
- 각 질의의 순위는 그 질의 문장과 후보 이미지만으로 계산한다. 다른 질의의 결과로 보정하지 않는다.
- 같은 문구의 임베딩은 재사용한다 (값이 문구만으로 정해지므로 다른 질의에 의존하지 않음).
- 재정렬·BLIP 사용 여부는 이미지 인코딩 직후, OWL 사용 여부는 OWL 단계 직전에 한 번만 정해
  모든 질의에 똑같이 적용한다 (시간 안전장치, 순위 보정이 아님).
- models/ 폴더의 가중치와 설정만 읽는다. 네트워크를 쓰지 않는다.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.functional import normalize
from transformers import AutoModel, AutoProcessor

from haiku.rerank import PhraseBank, all_phrases, condition_margins, extract_conditions, rerank_scores

IMG_BATCH, TXT_BATCH, MAX_TEXT_LEN, BLIP_BATCH, ITM_QBATCH, OWL_BATCH = 64, 256, 64, 64, 128, 32
WORKERS = int(os.environ.get("HAIKU_WORKERS", min(4, os.cpu_count() or 1)))


def load_model(model_dir, device):
    root = Path(model_dir) / "models"
    device = torch.device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    net = AutoModel.from_pretrained(root / "siglip", local_files_only=True, dtype=dtype).to(device).eval()
    proc = AutoProcessor.from_pretrained(root / "siglip", local_files_only=True)
    cfg = json.loads((root / "haiku_config.json").read_text(encoding="utf-8"))
    model = {"net": net, "proc": proc, "device": device, "cfg": cfg, "blip": None}
    if (cfg.get("itm") or {}).get("use") and (root / "blip").exists():
        from transformers import BlipForImageTextRetrieval
        blip = BlipForImageTextRetrieval.from_pretrained(root / "blip", local_files_only=True, dtype=dtype).to(device).eval()
        s0p = root / "blip" / "s0_weights.safetensors"                          # 팀원 형제 오답 학습 가중치(S0)를 공개 BLIP 위에 덮어씀
        if s0p.exists():
            from safetensors.torch import load_file
            sd, msd = load_file(str(s0p)), blip.state_dict()
            def _nk(k):
                for pre in ("module.", "blip.", "model."):
                    if k.startswith(pre) and k[len(pre):] in msd:
                        return k[len(pre):]
                return k
            sd = {_nk(k): v for k, v in sd.items()}
            hit = [k for k, v in sd.items() if k in msd and msd[k].shape == v.shape]
            if not hit or len(hit) < 0.9 * len(sd):
                raise RuntimeError("S0 가중치 키가 BLIP과 맞지 않음: 일치 %d / %d" % (len(hit), len(sd)))
            blip.load_state_dict({k: sd[k].to(msd[k].dtype) for k in hit}, strict=False)
            model["s0_keys"] = len(hit)
        model["blip"] = blip
        model["blip_proc"] = AutoProcessor.from_pretrained(root / "blip", local_files_only=True)
    model["net_zs"] = None
    if (root / "siglip_zs").exists():                                         # 색 검사용 제로샷 SigLIP (파인튜닝 전 가중치)
        model["net_zs"] = AutoModel.from_pretrained(root / "siglip_zs", local_files_only=True, dtype=dtype).to(device).eval()
    model["owl"] = None
    if (cfg.get("owl") or {}).get("use") and (root / "owl").exists():
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
        model["owl"] = OwlViTForObjectDetection.from_pretrained(root / "owl", local_files_only=True, dtype=dtype).to(device).eval()
        model["owl_proc"] = OwlViTProcessor.from_pretrained(root / "owl", local_files_only=True)
    return model


def _size(ip):
    """전처리 설정의 입력 크기 (dict / SizeDict / 숫자 모두 처리)."""
    s = ip.size
    for key in ("height", "shortest_edge"):
        v = getattr(s, key, None)
        if v is None and isinstance(s, dict):
            v = s.get(key)
        if v:
            return int(v)
    return int(s) if isinstance(s, (int, float)) else 384


class _Images(torch.utils.data.Dataset):
    def __init__(self, paths, image_processor):
        self.paths, self.ip = paths, image_processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
        return self.ip(images=im, return_tensors="pt")["pixel_values"][0]


class _Raw(torch.utils.data.Dataset):
    """같은 크기로 줄인 uint8 이미지 (SigLIP·BLIP이 함께 씀)."""

    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)


def _feats(out):
    return out.pooler_output if hasattr(out, "pooler_output") else out


def _norm(x_uint8, ip, size, dtype):
    x = x_uint8.float() / 255.0
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
    mean = torch.tensor(ip.image_mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std, device=x.device).view(1, 3, 1, 1)
    return ((x - mean) / std).to(dtype)


@torch.inference_mode()
def encode_images(model, paths):
    net, dev = model["net"], model["device"]
    dl = torch.utils.data.DataLoader(_Images(paths, model["proc"].image_processor), batch_size=IMG_BATCH,
                                     num_workers=WORKERS, pin_memory=dev.type == "cuda")
    return torch.cat([normalize(_feats(net.get_image_features(
        pixel_values=px.to(dev, dtype=net.dtype, non_blocking=True))).float(), dim=-1) for px in dl])


@torch.inference_mode()
def encode_images_raw(model, paths):
    """이미지를 한 번만 읽어 SigLIP 임베딩과 BLIP용 uint8 보관본을 함께 만든다."""
    net, dev = model["net"], model["device"]
    sip, bip = model["proc"].image_processor, model["blip_proc"].image_processor
    s_size, b_size = _size(sip), _size(bip)
    dl = torch.utils.data.DataLoader(_Raw(paths, b_size), batch_size=IMG_BATCH, num_workers=WORKERS,
                                     pin_memory=dev.type == "cuda")
    raw = torch.empty((len(paths), 3, b_size, b_size), dtype=torch.uint8, device=dev)
    vecs, s = [], 0
    for x in dl:
        x = x.to(dev, non_blocking=True)
        raw[s:s + len(x)] = x
        s += len(x)
        vecs.append(normalize(_feats(net.get_image_features(pixel_values=_norm(x, sip, s_size, net.dtype))).float(), dim=-1))
    return torch.cat(vecs), raw


@torch.inference_mode()
def encode_texts(model, prompts, key="net"):
    net, proc, dev = model.get(key) or model["net"], model["proc"], model["device"]
    out = []
    for s in range(0, len(prompts), TXT_BATCH):
        tok = proc(text=prompts[s:s + TXT_BATCH], padding="max_length", max_length=MAX_TEXT_LEN,
                   truncation=True, return_tensors="pt").to(dev)
        out.append(normalize(_feats(net.get_text_features(**tok)).float(), dim=-1))
    return torch.cat(out)


@torch.inference_mode()
def itm_scores(model, raw, img_idx, texts):
    """(문장, 이미지) 쌍마다 BLIP 매칭 로짓 차이. 이미지 특징은 이번 묶음에 필요한 것만 계산."""
    blip, bproc, dev = model["blip"], model["blip_proc"], model["device"]
    bip = bproc.image_processor
    uniq = sorted(set(img_idx))
    where = {j: n for n, j in enumerate(uniq)}
    feats = []
    for s in range(0, len(uniq), BLIP_BATCH):
        px = _norm(raw[uniq[s:s + BLIP_BATCH]], bip, _size(bip), next(blip.parameters()).dtype)
        feats.append(blip.vision_model(pixel_values=px).last_hidden_state)
    feats = torch.cat(feats)
    out = []
    for s in range(0, len(texts), BLIP_BATCH):
        tok = bproc.tokenizer(texts[s:s + BLIP_BATCH], padding=True, truncation=True, max_length=128,
                              return_tensors="pt").to(dev)
        img = feats[[where[j] for j in img_idx[s:s + BLIP_BATCH]]]
        att = torch.ones(img.shape[:-1], dtype=torch.long, device=dev)
        h = blip.text_encoder(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"],
                              encoder_hidden_states=img, encoder_attention_mask=att).last_hidden_state
        logit = blip.itm_head(h[:, 0, :]).float()
        out.append(logit[:, 1] - logit[:, 0])
    return torch.cat(out)


@torch.inference_mode()
def owl_detect(model, paths, need, deadline=None):
    """need: {후보 번호: {물체 이름}} → {(후보 번호, 이름): (검출 점수, cx, cy, w, h)}.
    이름마다 점수가 가장 높은 상자 하나 (haiku.owl.OwlChecker.detect와 같은 계산, 이미지는 한 번만 읽음)."""
    owl, proc, dev = model["owl"], model["owl_proc"], model["device"]
    names = sorted({n for v in need.values() for n in v})
    row = {n: k for k, n in enumerate(names)}
    tok = proc.tokenizer([f"a photo of a {n}" for n in names] + [" "], padding="max_length", truncation=True,
                         return_tensors="pt").to(dev)
    pad = len(names)
    js = sorted(need)
    dl = torch.utils.data.DataLoader(_Images([paths[j] for j in js], proc.image_processor), batch_size=OWL_BATCH,
                                     num_workers=WORKERS, pin_memory=dev.type == "cuda")
    out, s = {}, 0
    for px in dl:
        if deadline is not None and time.perf_counter() > deadline:
            break                                                          # 경과 시간 기준 중단: 남은 이미지는 OWL 결과 없음(순서 유지)
        chunk = js[s:s + len(px)]
        s += len(px)
        lists = [sorted(need[j]) for j in chunk]
        width = max(len(l) for l in lists)
        ridx = torch.tensor([[row[n] for n in l] + [pad] * (width - len(l)) for l in lists], device=dev).view(-1)
        o = owl(input_ids=tok["input_ids"][ridx], attention_mask=tok["attention_mask"][ridx],
                pixel_values=px.to(dev, dtype=next(owl.parameters()).dtype, non_blocking=True))
        probs, boxes = o.logits.float().sigmoid(), o.pred_boxes.float()
        best = probs.argmax(dim=1)                                          # (B, 질의 수)
        for b, (j, l) in enumerate(zip(chunk, lists)):
            for q, n in enumerate(l):
                p = int(best[b, q])
                out[(j, n)] = (float(probs[b, p, q]), float(boxes[b, p, 0]), float(boxes[b, p, 1]),
                               float(boxes[b, p, 2]), float(boxes[b, p, 3]))
    return out


class _Crops(torch.utils.data.Dataset):
    """OWL 상자를 조금 넓혀 잘라 SigLIP 입력으로 만든다 (DataLoader worker에서 병렬 처리)."""

    def __init__(self, items, image_processor, pad):
        self.items, self.ip, self.pad = items, image_processor, pad

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, (_, cx, cy, w, h) = self.items[i]
        with Image.open(path) as im:
            im = im.convert("RGB")
            W, H = im.size
            x0, x1 = (cx - w / 2 - self.pad * w) * W, (cx + w / 2 + self.pad * w) * W
            y0, y1 = (cy - h / 2 - self.pad * h) * H, (cy + h / 2 + self.pad * h) * H
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(W, max(int(x1), x0 + 8)), min(H, max(int(y1), y0 + 8))
            im = im.crop((x0, y0, x1, y1))
        return self.ip(images=im, return_tensors="pt")["pixel_values"][0]


@torch.inference_mode()
def crop_margins(model, paths, det, keys, texts, pad):
    """keys: [(후보 번호, 물체 이름)] → 잘라 본 SigLIP 벡터. texts: 문구 → 벡터 (이미 계산된 것)."""
    net, dev = model.get("net_zs") or model["net"], model["device"]
    dl = torch.utils.data.DataLoader(_Crops([(paths[j], det[(j, n)]) for j, n in keys], model["proc"].image_processor, pad),
                                     batch_size=IMG_BATCH, num_workers=WORKERS, pin_memory=dev.type == "cuda")
    vecs = torch.cat([normalize(_feats(net.get_image_features(pixel_values=px.to(dev, dtype=net.dtype, non_blocking=True))).float(),
                                dim=-1) for px in dl])
    return {k: v for k, v in zip(keys, vecs)}


def _pick_front(seq, gains):
    """gains[j] = 후보 j가 1위보다 나은 정도. 가장 큰 후보(동률이면 앞쪽)가 1위보다 낫다면 맨 앞으로, 나머지는 원래 순서."""
    best = max(range(len(gains)), key=lambda j: (gains[j], -j))
    if best > 0 and gains[best] > 0:
        seq = list(seq); seq.insert(0, seq.pop(best))
        return seq, True
    return list(seq), False


def owl_rerank(model, queries, paths, orders, heads, cfg, t0):
    """최종 상위 2장만 OWL 상자로 다시 본다 (순서 교환만).
    (a) 물체별 색 검사: 상자를 잘라 SigLIP으로 '목표 색 vs 같은 질의의 다른 물체 색'을 비교, 원래 점수에 mu를 곱해 더함.
    (b) 관계 규칙: 좌우·상하(+선택: 위·안·옆) 관계를 상자로 확인.
    같은 이미지의 검출은 한 번만 계산한다. 사용 여부는 경과 시간만으로 모든 질의에 똑같이 정한다."""
    from haiku.colorcrop import crop_conditions
    from haiku.owl import BASE_RELS, relation_score, spatial_relations
    ocfg, ccfg = cfg.get("owl") or {}, cfg.get("color") or {}
    kinds = tuple(ocfg.get("kinds") or BASE_RELS)
    use_color = bool(ccfg.get("use"))
    topn = max(2, int(ocfg.get("top_n", 2)))                                  # 상자로 다시 볼 상위 후보 수 (기본 2: 1·2위 교환)
    rels, crels, need = {}, {}, {}
    for n, q in enumerate(queries):
        if len(orders[n]) < 2:
            continue
        r = spatial_relations(q["prompt"], kinds)
        cc = crop_conditions(q["prompt"]) if use_color else []
        names = set()
        if r:
            rels[n] = r
            names |= {x for _, a, b in r for x in (a, b)}
        if cc:
            crels[n] = cc
            names |= {nm for nm, _, _ in cc}
        for j in orders[n][:topn] if names else ():
            need.setdefault(j, set()).update(names)
    info = {"owl_queries": len(rels), "color_queries": len(crels), "owl_images": len(need), "owl_used": False}
    elapsed = time.perf_counter() - t0
    info.update(owl_start_sec=round(elapsed, 1))
    if not need or elapsed > float(ocfg.get("start_before_sec", 600)):
        return info
    det = owl_detect(model, paths, need, deadline=t0 + float(ocfg.get("stop_after_sec", 840)))
    info.update(owl_used=True, owl_det=len(det), owl_sec=round(time.perf_counter() - t0 - elapsed, 1))
    # (a) 색 검사
    swapped_c = 0
    if crels and time.perf_counter() - t0 < float(ccfg.get("start_before_sec", 840)):
        cthr, mu, pad = float(ccfg.get("thr", 0.1)), float(ccfg.get("mu", 2.0)), float(ccfg.get("pad", 0.1))
        keys, phrases = {}, set()
        for n, cc in crels.items():
            for j in orders[n][:topn]:
                for nm, tgt, cons in cc:
                    if (j, nm) in det and det[(j, nm)][0] >= cthr:
                        keys[(j, nm)] = None
            for _, tgt, cons in cc:
                phrases.update([tgt] + cons)
        phrases = sorted(phrases)
        if keys:
            tv = dict(zip(phrases, encode_texts(model, phrases, "net_zs")))
            cv = crop_margins(model, paths, det, list(keys), tv, pad)

            def cmargin(j, cc):
                out = []
                for nm, tgt, cons in cc:
                    if (j, nm) in cv:
                        T, C = tv[tgt], torch.stack([tv[c] for c in cons])
                        out.append((nm, float(cv[(j, nm)] @ T - (C @ cv[(j, nm)]).max())))
                return dict(out)
            for n, cc in crels.items():
                o = orders[n]
                m0 = cmargin(o[0], cc)
                gains = [0.0]
                for k in range(1, min(topn, len(o))):
                    mk = cmargin(o[k], cc)
                    both = [nm for nm, _, _ in cc if nm in m0 and nm in mk]    # 두 후보 모두에서 물체를 찾은 조건만 사용
                    gains.append((float(heads[n][k]) + mu * sum(mk[x] for x in both)) -
                                 (float(heads[n][0]) + mu * sum(m0[x] for x in both)) if both else -1.0)
                orders[n], moved = _pick_front(o, gains)
                swapped_c += moved
        info.update(color_swapped=swapped_c, color_sec=round(time.perf_counter() - t0 - elapsed - info["owl_sec"], 1))
    # (b) 관계 규칙: 색 검사 뒤의 상위 후보 (검출 재사용)
    thr, margin = float(ocfg.get("thr", 0.1)), float(ocfg.get("margin", 0.06))
    swapped = 0
    for n, r in rels.items():
        names = {x for _, a, b in r for x in (a, b)}
        top = orders[n][:topn]
        if any((j, x) not in det for j in top for x in names):
            continue                                                       # 시간 안에 검출하지 못한 질의는 순서 유지
        sc = [relation_score({x: det[(j, x)] for x in names}, r, thr, margin) for j in top]
        new, moved = _pick_front(orders[n], [v - sc[0] for v in sc])
        orders[n] = new
        swapped += moved
    info.update(owl_swapped=swapped)
    return info


@torch.inference_mode()
def predict(model, queries, candidates, top_k=5):
    t0 = time.perf_counter()
    cfg = model["cfg"]
    K, method, lam = int(cfg.get("K", 10)), cfg.get("method", "B"), float(cfg.get("lam", 0.0))
    reliability, tau = cfg.get("reliability", {}), cfg.get("tau")
    kinds = set(cfg["kinds"]) if cfg.get("kinds") else None          # 쓸 조건 종류 (없으면 전부)
    icfg = cfg.get("itm") or {}
    image_ids = [c["image_id"] for c in candidates]
    paths = [c["image_path"] for c in candidates]
    if model.get("blip") is not None:
        image_vecs, raw = encode_images_raw(model, paths)
    else:
        image_vecs, raw = encode_images(model, paths), None
    t_img = time.perf_counter() - t0
    timing = {"encode_sec": round(t_img, 1)}
    # 재정렬·BLIP 사용 여부는 여기서 한 번만 정한다 (질의 내용과 무관, 모든 질의에 동일)
    elapsed = time.perf_counter() - t0
    use_rerank = method != "A" and lam > 0 and elapsed < float(cfg.get("skip_rerank_after_sec", 700))
    use_itm = raw is not None and icfg.get("use") and elapsed < float(icfg.get("skip_after_sec", 600))
    k_itm, lam_itm, tau_itm = int(icfg.get("k", 2)), float(icfg.get("lam", 0.0)), icfg.get("tau")
    bank = PhraseBank(lambda t: encode_texts(model, t))
    orders, heads = [], []
    for s in range(0, len(queries), TXT_BATCH):
        batch = queries[s:s + TXT_BATCH]
        text_vecs = encode_texts(model, [q["prompt"] for q in batch])
        top = (text_vecs @ image_vecs.T).topk(max(K, top_k), dim=1)
        conds = [[c for c in extract_conditions(q["prompt"]) if kinds is None or c["kind"] in kinds]
                 for q in batch] if use_rerank else None
        if use_rerank:
            bank.add([p for c in conds for p in all_phrases(c)])
        for i in range(len(batch)):
            idx, s_k = top.indices[i], top.values[i]
            new = s_k[:K]
            if use_rerank and conds[i]:
                M = condition_margins(conds[i], bank, image_vecs[idx[:K]])
                new = rerank_scores(s_k[:K], conds[i], M, method, lam, reliability, tau)
            o = new.argsort(descending=True)
            orders.append(torch.cat([idx[:K][o], idx[K:]]).tolist())
            heads.append(new[o].float())
    timing["rerank_end_sec"] = round(time.perf_counter() - t0, 1)
    if use_itm and lam_itm > 0:
        k_itm = min(k_itm, K)
        sel = [n for n in range(len(queries)) if tau_itm is None or float(heads[n][0] - heads[n][1]) < tau_itm]
        # 같은 후보를 보는 질의끼리 묶어 BLIP 이미지 특징 재계산을 줄인다 (각 질의의 결과는 묶음과 무관)
        sel.sort(key=lambda n: orders[n][0])
        for s in range(0, len(sel), ITM_QBATCH):
            chunk = sel[s:s + ITM_QBATCH]
            texts = [queries[n]["prompt"] for n in chunk for _ in range(k_itm)]
            img_idx = [orders[n][j] for n in chunk for j in range(k_itm)]
            sc = itm_scores(model, raw, img_idx, texts).view(len(chunk), k_itm)
            for r, n in enumerate(chunk):
                h = heads[n][:k_itm].to(sc) + lam_itm * sc[r]
                srt = h.argsort(descending=True)
                head = [orders[n][j] for j in srt.tolist()]
                orders[n] = head + orders[n][k_itm:]
                heads[n] = torch.cat([h[srt].float(), heads[n][k_itm:]])      # 색 검사가 쓸 현재 점수
    timing["itm_used"] = bool(use_itm and lam_itm > 0)
    timing["itm_end_sec"] = round(time.perf_counter() - t0, 1)
    ocfg = cfg.get("owl") or {}
    if model.get("owl") is not None and ocfg.get("use"):
        timing.update(owl_rerank(model, queries, paths, orders, heads, cfg, t0))
    timing["total_sec"] = round(time.perf_counter() - t0, 1)
    model["last_timing"] = timing
    return {q["prompt_id"]: [image_ids[j] for j in orders[n][:top_k]] for n, q in enumerate(queries)}
