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

OUT = open(os.environ.get("V_OUT", "/content/mix_result.txt"), "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()

EPOCHS, BS, LR, NL = int(os.environ.get("V_EP", 6)), 32, 2e-5, int(os.environ.get("V_NL", 8))
DATA = os.environ.get("V_DATA", "/content/haikuthon-train"); dev = torch.device("cuda")
rows = load_rows(DATA); by_id = {r["id"]: r for r in rows}
parsed = {r["id"]: parse_prompt(r["prompt"]) for r in rows}
group_of, members = build_groups(rows, parsed)
tr, va = group_split(rows, members, 42)
allids = [r["id"] for r in rows]; pos = {i: k for k, i in enumerate(allids)}
P("그룹 분할: 학습 %d / 검증 %d" % (len(tr), len(va)))
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


# ================= 섞기: 파인튜닝 + (조건B) + 색 + OWL + 학습형 =================
net.load_state_dict(best[1]); P("\n[섞기] 파인튜닝 에포크 %d 사용" % best[2])
F_img = enc_img(net, vi); F_txt = enc_txt(net, vtexts); tf_f = lambda t: enc_txt(net, t)
del opt, scaler; torch.cuda.empty_cache()
K = 10
cond_v = {i: extract_conditions(by_id[i]["prompt"]) for i in va}
bank = PhraseBank(tf_f); bank.add([p for i in va for p in all_phrases(cond_v[i])])
def make_items(lam):
    items = []
    for k, q in enumerate(va):
        s = F_txt[k] @ F_img.T; top = s.topk(K); new = top.values
        if lam > 0 and cond_v[q]:
            new = rerank_scores(top.values, cond_v[q], condition_margins(cond_v[q], bank, F_img[top.indices]), "B", lam, {})
        o = new.argsort(descending=True)
        order = [va[j] for j in top.indices[o].tolist()]
        items.append({"q": q, "order": order, "sc": new[o].float().cpu().tolist(), "gold": order.index(q) + 1 if q in order else K + 1})
    return items
scr = lambda rk: float(np.mean([point(r) for r in rk])); r1f = lambda rk: float(np.mean([r == 1 for r in rk]))
def rank_after(it, o): return o.index(it["q"]) + 1 if it["q"] in o[:5] else (6 if it["q"] in o else K + 1)
def gl(items, orders):
    g = l = 0
    for i, o in zip(items, orders):
        a, b = i["gold"], rank_after(i, o); g += (b < a); l += (b > a)
    return g, l
owl = OwlChecker(device="cuda"); cc = CropColor(owl, device="cuda", kinds=("color_x",)); ks = ("color_x",)
def pick_front(o, vals):
    best_j = max(range(len(vals)), key=lambda j: (vals[j], -j))
    if best_j > 0 and vals[best_j] > 0:
        o = list(o); x = o.pop(best_j); o.insert(0, x)
    return list(o)
_MC = {}
def cm(todo):
    miss = [t for t in dict.fromkeys(todo) if t not in _MC]
    if miss:
        for t, m in zip(miss, cc.margins(miss)): _MC[t] = m
    return {t: _MC[t] for t in todo}
def owl_orders(items, thr, mg, nn, order_in):
    need, rels = [], {}
    for it in items:
        r = spatial_relations(by_id[it["q"]]["prompt"], EXT_RELS); rels[it["q"]] = r
        o = order_in[it["q"]]
        if r and len(o) >= 2:
            it["_n"] = tuple(sorted({a for _, x, y in r for a in (x, y)})); need += [(by_id[g]["image_path"], it["_n"]) for g in o[:nn]]
    owl.detect(need); out = []
    for it in items:
        o = list(order_in[it["q"]]); r = rels[it["q"]]
        if r and len(o) >= 2:
            ss = [relation_score(owl.cache[(by_id[g]["image_path"], it["_n"])], r, thr, mg) for g in o[:nn]]
            o = pick_front(o, [x - ss[0] for x in ss])
        out.append(o)
    return out
def color_orders(items, mu, nn, order_in):
    todo = []
    for it in items:
        o = order_in[it["q"]]
        if crop_conditions(by_id[it["q"]]["prompt"], ks) and len(o) >= 2: todo += [(by_id[g]["image_path"], by_id[it["q"]]["prompt"]) for g in o[:nn]]
    mm = cm(todo); outs = []
    for it in items:
        o = list(order_in[it["q"]]); p = by_id[it["q"]]["prompt"]
        if crop_conditions(p, ks) and len(o) >= 2:
            # 점수는 현재 순서의 원 점수(it["sc"])를 쓰되 순서가 같은 경우만 유효
            vals = [0.0]
            for j in range(1, min(nn, len(o))):
                sa, sb, used = pair_scores(mm[(by_id[o[0]]["image_path"], p)], mm[(by_id[o[j]]["image_path"], p)], "crop")
                s0 = it["sc"][it["order"].index(o[0])]; sj = it["sc"][it["order"].index(o[j])]
                vals.append((sj + mu * sb) - (s0 + mu * sa) if used else -1.0)
            o = pick_front(o, vals)
        outs.append(o)
    return outs
def ev(items, orders): rk = [rank_after(i, o) for i, o in zip(items, orders)]; return scr(rk), r1f(rk)

for lam in (0.0, 1.0, 3.2):
    items = make_items(lam); base_o = {i["q"]: i["order"] for i in items}
    P("\n[기반: 파인튜닝 + 조건B lam=%.1f] 기준 점수 %.4f R@1 %.4f" % (lam, scr([i["gold"] for i in items]), r1f([i["gold"] for i in items])))
    for mu in (0.5, 1.0, 2.0, 5.0, 10.0):
        co = color_orders(items, mu, 3, base_o); cd = {i["q"]: o for i, o in zip(items, co)}
        s_c = ev(items, co)
        s_ce = ev(items, owl_orders(items, 0.1, 0.03, 3, cd))
        P("  색 mu=%-4.1f  색만 %.4f | 색+OWL(E규칙) %.4f R@1 %.4f | 색+OWL 개선/악화 %d/%d" % (mu, s_c[0], s_ce[0], s_ce[1], *gl(items, owl_orders(items, 0.1, 0.03, 3, cd))))
    s_o = ev(items, owl_orders(items, 0.1, 0.03, 3, base_o)); P("  OWL EXT만 %.4f" % s_o[0])

# ---------- 학습형 재순위 (검증 800 안에서 5-fold CV) ----------
N = 5; THR, MG = 0.1, 0.03
def feats(items):
    need, rels = [], {}
    for it in items:
        r = spatial_relations(by_id[it["q"]]["prompt"], EXT_RELS); rels[it["q"]] = r
        if r: it["_n"] = tuple(sorted({a for _, x, y in r for a in (x, y)})); need += [(by_id[g]["image_path"], it["_n"]) for g in it["order"][:N]]
    owl.detect(need); todo = []
    for it in items:
        p = by_id[it["q"]]["prompt"]
        if crop_conditions(p, ks): todo += [(by_id[g]["image_path"], p) for g in it["order"][:N]]
    mm = cm(todo); X = []; Y = []
    for it in items:
        o = it["order"][:N]; sc = it["sc"]; p = by_id[it["q"]]["prompt"]; r = rels[it["q"]]
        hasc = bool(crop_conditions(p, ks)); hasr = bool(r)
        rs = [relation_score(owl.cache[(by_id[g]["image_path"], it["_n"])], r, THR, MG) for g in o] if hasr else [0] * len(o)
        rows_ = []
        for j in range(len(o)):
            if hasc and j > 0:
                sa, sb, used = pair_scores(mm[(by_id[o[0]]["image_path"], p)], mm[(by_id[o[j]]["image_path"], p)], "crop"); col = (sb - sa) if used else 0.0; cu = 1.0 if used else 0.0
            else: col = 0.0; cu = 0.0
            d = sc[j] - sc[0]
            rows_.append([d, col, cu, rs[j] - rs[0], float(hasr), float(hasc), float(j == 1), float(j == 2), float(j == 3), float(j == 4), d * float(hasc), d * float(hasr)])
        while len(rows_) < N: rows_.append([-9.0] + [0.0] * 11)
        X.append(rows_); Y.append(o.index(it["q"]) if it["q"] in o else -1)
    return np.array(X, dtype=np.float32), np.array(Y)
def fit_clogit(X, Y, l2=1e-2):
    m = Y >= 0; Xt = torch.tensor(X[m]); Yt = torch.tensor(Y[m]); w = torch.zeros(X.shape[2], requires_grad=True)
    o_ = torch.optim.LBFGS([w], max_iter=300, line_search_fn="strong_wolfe")
    def cl():
        o_.zero_grad(); loss = F.cross_entropy(Xt @ w, Yt) + l2 * (w ** 2).sum(); loss.backward(); return loss
    o_.step(cl); return w.detach().numpy()
for lam in (0.0, 1.0):
    items = make_items(lam); X, Y = feats(items); idx = np.arange(len(items)); rs_ = np.random.RandomState(0); rs_.shuffle(idx); folds = np.array_split(idx, 5)
    for l2 in (1e-3, 1e-2, 1e-1):
        S = np.zeros((len(items), N))
        for f in folds:
            tr_i = np.setdiff1d(idx, f); w = fit_clogit(X[tr_i], Y[tr_i], l2); S[f] = X[f] @ w
        for rsz in (3, 5):
            orders = []
            for it, s in zip(items, S):
                o = list(it["order"]); k = min(rsz, len(o)); ix = sorted(range(k), key=lambda j: (-s[j], j)); orders.append([o[j] for j in ix] + o[k:])
            sc_, r1_ = ev(items, orders); g_, l_ = gl(items, orders)
            P("학습형(CV5) lam=%.1f l2=%g top%d | 점수 %.4f R@1 %.4f | 개선 %d 악화 %d" % (lam, l2, rsz, sc_, r1_, g_, l_))
P("\n끝")
