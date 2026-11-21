"""SigLIP-384 백본 파인튜닝(마지막 8층, 형제 그룹 어려운 오답 배치) 검증.
그룹 분할(학습 3200 / 검증 800). 비교: 제로샷 vs 파인튜닝 vs 융합, 풀 800 / 풀 4000, 서로 다른 1위 수(쏠림 점검), 조건 재정렬 결합."""
import json, math, os, random, sys, time, warnings
ROOT = os.environ.get("V_ROOT", "/content/v5"); sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from transformers import AutoModel, AutoProcessor
from haiku.data import load_rows
from haiku.siblings import build_groups, group_split, parse_prompt
from haiku.combo import SIGLIP_ID
from haiku.owl import OwlChecker, spatial_relations, relation_score, EXT_RELS
from haiku.colorcrop import CropColor, crop_conditions, pair_scores
from haiku.rerank import PhraseBank, all_phrases, condition_margins, extract_conditions, rerank_scores

OUT = open(os.environ.get("V_OUT", "/content/ftall_result.txt"), "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()

EPOCHS, BS, LR, NL = int(os.environ.get("V_EP", 6)), 32, 2e-5, int(os.environ.get("V_NL", 8))
DATA = os.environ.get("V_DATA", "/content/haikuthon-train"); dev = torch.device("cuda")
rows = load_rows(DATA); by_id = {r["id"]: r for r in rows}
parsed = {r["id"]: parse_prompt(r["prompt"]) for r in rows}
group_of, members = build_groups(rows, parsed)
tr, va = group_split(rows, members, 42)
if os.environ.get('V_ALL', '1') == '1': tr = list(tr) + list(va)   # V_ALL=1: 4000쌍 전부 학습(검증 누수), 0: 3200쌍만(정직한 검증)
allids = [r["id"] for r in rows]; pos = {i: k for k, i in enumerate(allids)}
P("학습 %d (검증 포함) / 검증 %d" % (len(tr), len(va)))
proc = AutoProcessor.from_pretrained(SIGLIP_ID)
f_ = lambda o: o.pooler_output if hasattr(o, "pooler_output") else o

def load_px(r):
    with Image.open(r["image_path"]) as im:
        return proc(images=im.convert("RGB"), return_tensors="pt").pixel_values[0].half()
t0 = time.time()
PX = torch.empty((len(rows), 3, 384, 384), dtype=torch.float16)
def _fill(k): PX[k] = load_px(rows[k])
with ThreadPoolExecutor(4) as ex: list(ex.map(_fill, range(len(rows))))
P("이미지 전처리 %.0f초 %s" % (time.time() - t0, tuple(PX.shape)))

def tok(texts): return proc(text=texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)

@torch.inference_mode()
def enc_img(net, idx):
    net.eval(); out = []
    for s in range(0, len(idx), 64):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(F.normalize(f_(net.get_image_features(pixel_values=PX[idx[s:s + 64]].to(dev))).float(), dim=-1))
    return torch.cat(out)

@torch.inference_mode()
def enc_txt(net, texts):
    net.eval(); out = []
    for s in range(0, len(texts), 128):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(F.normalize(f_(net.get_text_features(**tok(texts[s:s + 128]))).float(), dim=-1))
    return torch.cat(out)

point = lambda r: 0.0 if r > 5 else 0.5 * ((r == 1) + 1 / math.log2(r + 1))
vi = [pos[i] for i in va]; vtexts = [by_id[i]["prompt"] for i in va]
trset = set(pos[i] for i in tr)

def metrics(S, gold_cols, cols_are_train=None, tag=""):
    S = S.float().cpu(); g = S[torch.arange(len(gold_cols)), torch.tensor(gold_cols)]
    rk = 1 + (S > g[:, None]).sum(1)
    sc = float(np.mean([point(int(r)) for r in rk])); r1 = float((rk == 1).float().mean())
    t1 = S.argmax(1).tolist(); uniq = len(set(t1))
    extra = ""
    if cols_are_train is not None: extra = " | 1위가 학습 이미지 %.1f%%" % (100 * np.mean([cols_are_train[c] for c in t1]))
    P("  %-26s 점수 %.4f R@1 %.4f | 서로 다른 1위 %d/%d%s" % (tag, sc, r1, uniq, len(t1), extra))
    return sc

# ---------- 제로샷 ----------
net = AutoModel.from_pretrained(SIGLIP_ID).to(dev)
allidx = list(range(len(rows))); t0 = time.time()
Z_img = enc_img(net, allidx); Z_txt = enc_txt(net, vtexts)
P("제로샷 인코딩 %.0f초" % (time.time() - t0))
S800_z = Z_txt @ Z_img[vi].T; S4000_z = Z_txt @ Z_img.T
is_tr = [pos_i in trset for pos_i in range(len(rows))]
P("[제로샷]"); metrics(S800_z, list(range(len(vi))), tag="풀 800"); metrics(S4000_z, vi, is_tr, tag="풀 4000")

# ---------- 조건 재정렬 (풀 800, K=10, B, lam 3.2) ----------
def rerank_val(img_v, txt_v, txt_fn):
    conds = {i: extract_conditions(by_id[i]["prompt"]) for i in va}
    bank = PhraseBank(txt_fn); bank.add([p for i in va for p in all_phrases(conds[i])])
    ranks = []
    for n, q in enumerate(va):
        s = txt_v[n] @ img_v.T; top = s.topk(10); new = top.values
        if conds[q]: new = rerank_scores(top.values, conds[q], condition_margins(conds[q], bank, img_v[top.indices]), "B", 3.2, {})
        o = top.indices[new.argsort(descending=True)].tolist()
        ranks.append(o.index(n) + 1 if n in o[:5] else 6)
    return float(np.mean([point(r) for r in ranks])), float(np.mean([r == 1 for r in ranks]))
tf_z = lambda t: enc_txt(net, t)
sc, r1 = rerank_val(Z_img[vi], Z_txt, tf_z); P("  제로샷 + 조건B(풀 800)        점수 %.4f R@1 %.4f" % (sc, r1))

# ---------- 파인튜닝 ----------
n = NL
for p in net.parameters(): p.requires_grad_(False)
mods = [net.vision_model.head, net.text_model.head, net.vision_model.post_layernorm, net.text_model.final_layer_norm]
mods += list(net.vision_model.encoder.layers[-n:]) + list(net.text_model.encoder.layers[-n:])
params = [p for m in mods for p in m.parameters()] + [net.logit_scale, net.logit_bias]
for p in params: p.requires_grad_(True)
net.float()
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01); scaler = torch.amp.GradScaler()
rng = random.Random(42)
trs = set(tr)
def make_batches():
    gl = [[i for i in members[g] if i in trs] for g in members]; gl = [g for g in gl if g]
    rng.shuffle(gl); flat = []
    for g in gl: rng.shuffle(g); flat += g
    return [flat[s:s + BS] for s in range(0, len(flat), BS)]
steps_total = EPOCHS * len(make_batches()); step = 0
best = (-1, None, 0)
for ep in range(1, EPOCHS + 1):
    net.train(); t0 = time.time(); tl = 0
    for b in make_batches():
        lr = LR * min(1, (step + 1) / 20) * 0.5 * (1 + math.cos(math.pi * step / steps_total)); step += 1
        for g in opt.param_groups: g["lr"] = lr
        px = PX[[pos[i] for i in b]].to(dev); tk = tok([by_id[i]["prompt"] for i in b])
        with torch.autocast("cuda", dtype=torch.float16):
            fi = F.normalize(f_(net.get_image_features(pixel_values=px)).float(), dim=-1)
            ft = F.normalize(f_(net.get_text_features(**tk)).float(), dim=-1)
        lg = fi @ ft.T * net.logit_scale.exp() + net.logit_bias
        lab = 2 * torch.eye(len(b), device=dev) - 1
        loss = -F.logsigmoid(lab * lg).sum() / len(b)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); tl += float(loss)
    Si = enc_img(net, vi); St = enc_txt(net, vtexts); S = St @ Si.T
    g = S.diagonal(); rk = 1 + (S > g[:, None]).sum(1).cpu()
    sc = float(np.mean([point(int(r)) for r in rk])); r1 = float((rk == 1).float().mean())
    P("에포크 %d | loss %.4f | 풀800 점수 %.4f R@1 %.4f | %.0f초" % (ep, tl, sc, r1, time.time() - t0))
    if sc > best[0]:
        best = (sc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, ep)
last_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}



from safetensors.torch import save_file
sd = {k: v.detach().cpu().half().contiguous() for k, v in net.state_dict().items()}
OUTW = os.environ.get("V_W", "/content/ft_all4000_fp16.safetensors")
save_file(sd, OUTW, metadata={"format": "pt"})
P("가중치 저장", OUTW, "%.0f MB" % (os.path.getsize(OUTW) / 1e6), "| 마지막 에포크 가중치 사용(학습 4000쌍 전부)")
P("끝")
