"""v4 위에 OWL 관계 확장 / 물체별 색 검사를 얹으면 좋아지는지: 그룹 분할 검증 (SigLIP-384 제로샷 + 조건 재정렬 B 기준)."""
import json, math, sys, time, warnings
import os
ROOT=os.environ.get("V_ROOT","/content/v5"); sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
import torch
from haiku.data import load_rows
from haiku.siblings import build_groups, group_split, parse_prompt
from haiku.combo import _siglip, SIGLIP_ID
from haiku.rerank import PhraseBank, all_phrases, condition_margins, extract_conditions, rerank_scores
from haiku.owl import OwlChecker, spatial_relations, relation_score, BASE_RELS, EXT_RELS
from haiku.colorcrop import CropColor, crop_conditions, pair_scores

OUT = open(os.environ.get("V_OUT","/content/e5_result.txt"), "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()

DATA, K, LAM = os.environ.get("V_DATA","/content/haikuthon-train"), 10, 3.2
dev_t = torch.device("cuda")
rows = load_rows(DATA); by_id = {r["id"]: r for r in rows}
parsed = {r["id"]: parse_prompt(r["prompt"]) for r in rows}
group_of, members = build_groups(rows, parsed)
tr, va = group_split(rows, members, 42)
P("그룹 분할: 학습 %d / 검증 %d" % (len(tr), len(va)))
img_fn, txt_fn = _siglip(SIGLIP_ID, dev_t)
ids = tr + va; pos = {i: k for k, i in enumerate(ids)}
t0 = time.time()
img = img_fn([by_id[i]["image_path"] for i in ids]); txt = txt_fn([by_id[i]["prompt"] for i in ids])
conds = {i: extract_conditions(by_id[i]["prompt"]) for i in ids}
bank = PhraseBank(txt_fn); bank.add([p for i in ids for p in all_phrases(conds[i])])
P("SigLIP 인코딩 %.0f초" % (time.time() - t0))

def base(q_ids, g_ids):
    G = img[[pos[g] for g in g_ids]]; items = []
    for q in q_ids:
        s = txt[pos[q]] @ G.T; top = s.topk(min(K, len(g_ids))); new = top.values
        if conds[q]:
            new = rerank_scores(top.values, conds[q], condition_margins(conds[q], bank, G[top.indices]), "B", LAM, {})
        o = new.argsort(descending=True)
        order = [g_ids[j] for j in top.indices[o].tolist()]
        sc = new[o].float().cpu().tolist()
        items.append({"q": q, "order": order, "sc": sc, "gold": order.index(q) + 1 if q in order else K + 1})
    return items

dev, val = base(tr, tr), base(va, va)
score = lambda rk: sum(0.5 * ((r == 1) + (1 / math.log2(r + 1) if r <= 5 else 0)) for r in rk) / len(rk)
r1 = lambda rk: sum(r == 1 for r in rk) / len(rk)
P("기준(SigLIP-384+조건B) dev %.4f / val %.4f R@1 %.4f" % (score([i["gold"] for i in dev]), score([i["gold"] for i in val]), r1([i["gold"] for i in val])))

owl = OwlChecker(device="cuda")
def rank_after(it, new_order):
    return new_order.index(it["q"]) + 1 if it["q"] in new_order[:5] else (6 if it["q"] in new_order else K + 1)

def report(name, items0, new_orders):
    base_rk = [i["gold"] for i in items0]; new_rk = [rank_after(i, o) for i, o in zip(items0, new_orders)]
    sw = sum(1 for i, o in zip(items0, new_orders) if o[:2] != i["order"][:2])
    P("  %-34s val %.4f R@1 %.4f | 1위 고침 %d 망침 %d | 바뀐 질의 %d" % (name, score(new_rk), r1(new_rk),
      sum(b > 1 and r == 1 for b, r in zip(base_rk, new_rk)), sum(b == 1 and r > 1 for b, r in zip(base_rk, new_rk)), sw))
    return score(new_rk)


def pick_front(o, vals):
    """vals[j] = 후보 j의 (점수 - 1위 점수) 이득. 이득이 가장 큰 후보(동률이면 앞쪽)를 맨 앞으로, 나머지는 원래 순서."""
    best = max(range(len(vals)), key=lambda j: (vals[j], -j))
    if best > 0 and vals[best] > 0:
        o = list(o); x = o.pop(best); o.insert(0, x)
    return list(o)

def owl_orders(items, kinds, thr, mg, n=2, order_in=None):
    need, rels = [], {}
    for it in items:
        r = spatial_relations(by_id[it["q"]]["prompt"], kinds); rels[it["q"]] = r
        o = (order_in or {}).get(it["q"], it["order"])
        if r and len(o) >= 2:
            names = tuple(sorted({a for _, x, y in r for a in (x, y)})); it["_n"] = names
            need += [(by_id[g]["image_path"], names) for g in o[:n]]
    owl.detect(need)
    out = []
    for it in items:
        o = list((order_in or {}).get(it["q"], it["order"])); r = rels[it["q"]]
        if r and len(o) >= 2:
            ss = [relation_score(owl.cache[(by_id[g]["image_path"], it["_n"])], r, thr, mg) for g in o[:n]]
            o = pick_front(o, [x - ss[0] for x in ss]) + []
        out.append(o)
    return out

cc = CropColor(owl, device="cuda", kinds=("color_x",)); ks = ("color_x",)

def color_orders(items, mu, n=2, order_in=None):
    todo = []
    for it in items:
        o = (order_in or {}).get(it["q"], it["order"])
        if crop_conditions(by_id[it["q"]]["prompt"], ks) and len(o) >= 2:
            todo += [(by_id[g]["image_path"], by_id[it["q"]]["prompt"]) for g in o[:n]]
    M = cc.margins(todo) if todo else []
    mm = {k: v for k, v in zip(todo, M)}
    outs = []
    for it in items:
        o = list((order_in or {}).get(it["q"], it["order"])); sc = it["sc"]
        p = by_id[it["q"]]["prompt"]
        if crop_conditions(p, ks) and len(o) >= 2:
            vals = [0.0]
            for j in range(1, min(n, len(o))):
                sa, sb, used = pair_scores(mm[(by_id[o[0]]["image_path"], p)], mm[(by_id[o[j]]["image_path"], p)], "crop")
                vals.append((sc[j] + mu * sb) - (sc[0] + mu * sa) if used else -1.0)
            o = pick_front(o, vals)
        outs.append(o)
    return outs




KE=EXT_RELS
def pick(kinds,items,nn,order_in=None):
    best=None
    for thr in (0.1,0.15):
        for mg in (0.03,0.06):
            s_=score([rank_after(i,o) for i,o in zip(items,owl_orders(items,kinds,thr,mg,nn,order_in))])
            if best is None or s_>best[0]: best=(s_,thr,mg)
    return best
def gl(items,new):
    g=l=0
    for i,o in zip(items,new):
        a=i["gold"];b=rank_after(i,o); g+=(b<a); l+=(b>a)
    return g,l
for nn in (3,4,5):
    t0=time.time(); res={}
    bm=None
    for m in (0.5,1.0,2.0,5.0):
        s_=score([rank_after(i,o) for i,o in zip(dev,color_orders(dev,m,nn))])
        if bm is None or s_>bm[0]: bm=(s_,m)
    mu=bm[1]
    cdev=dict(zip([i["q"] for i in dev],color_orders(dev,mu,nn))); cval=dict(zip([i["q"] for i in val],color_orders(val,mu,nn)))
    bE=pick(KE,dev,nn,cdev)
    out={}
    for nm,items,cv in (("dev",dev,cdev),("val",val,cval)):
        orders=owl_orders(items,KE,bE[1],bE[2],nn,cv)
        out[nm]=(score([rank_after(i,o) for i,o in zip(items,orders)]),gl(items,orders))
    P("n=%d | mu %.1f thr %.2f mg %.2f | dev %.4f val %.4f | val 개선 %d 악화 %d | %.0f초"%(nn,mu,bE[1],bE[2],out["dev"][0],out["val"][0],out["val"][1][0],out["val"][1][1],time.time()-t0))
P("끝")
