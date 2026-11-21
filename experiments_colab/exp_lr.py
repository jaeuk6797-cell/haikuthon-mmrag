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

OUT = open(os.environ.get("V_OUT","/content/lr_result.txt"), "w")
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




import numpy as np
N=int(os.environ.get("V_N","5")); THR,MG=0.1,0.03
def feats(items):
    """질의별 상위 N 후보 특징 행렬 [N,F] (0번 후보 기준 차이)."""
    # OWL 관계
    need=[];rels={}
    for it in items:
        p=by_id[it["q"]]["prompt"]; r=spatial_relations(p,EXT_RELS); rels[it["q"]]=r
        if r:
            nm=tuple(sorted({a for _,x,y in r for a in (x,y)})); it["_n"]=nm
            need+=[(by_id[g]["image_path"],nm) for g in it["order"][:N]]
    owl.detect(need)
    todo=[]
    for it in items:
        p=by_id[it["q"]]["prompt"]
        if crop_conditions(p,ks): todo+=[(by_id[g]["image_path"],p) for g in it["order"][:N]]
    M=cc.margins(todo) if todo else []; mm=dict(zip(todo,M))
    X=[];Y=[]
    for it in items:
        o=it["order"][:N]; sc=it["sc"]; p=by_id[it["q"]]["prompt"]; r=rels[it["q"]]
        hasc=bool(crop_conditions(p,ks)); hasr=bool(r)
        rs=[relation_score(owl.cache[(by_id[g]["image_path"],it["_n"])],r,THR,MG) for g in o] if hasr else [0]*len(o)
        rows_=[]
        for j in range(len(o)):
            if hasc and j>0:
                sa,sb,used=pair_scores(mm[(by_id[o[0]]["image_path"],p)],mm[(by_id[o[j]]["image_path"],p)],"crop")
                col=(sb-sa) if used else 0.0; cu=1.0 if used else 0.0
            else: col=0.0;cu=0.0
            rows_.append([sc[j]-sc[0],col,cu,rs[j]-rs[0],float(hasr),float(hasc),float(j==1),float(j==2),float(j==3),float(j==4),
                          (sc[j]-sc[0])*float(hasc),(sc[j]-sc[0])*float(hasr)])
        while len(rows_)<N: rows_.append([-9.0]+[0.0]*11)
        X.append(rows_); Y.append(o.index(it["q"]) if it["q"] in o else -1)
    return np.array(X,dtype=np.float32),np.array(Y)
t0=time.time(); Xd,Yd=feats(dev); Xv,Yv=feats(val); P("특징 추출 %.0f초 | X %s"%(time.time()-t0,Xd.shape))
def fit_clogit(X,Y,l2=1e-2,steps=300):
    m=Y>=0; Xt=torch.tensor(X[m]);Yt=torch.tensor(Y[m]); w=torch.zeros(X.shape[2],requires_grad=True)
    opt=torch.optim.LBFGS([w],max_iter=steps,line_search_fn="strong_wolfe")
    def cl():
        opt.zero_grad(); lo=(Xt@w); loss=torch.nn.functional.cross_entropy(lo,Yt)+l2*(w**2).sum(); loss.backward(); return loss
    opt.step(cl); return w.detach().numpy()
def orders_from(items,S,restrict):
    out=[]
    for it,s in zip(items,S):
        o=list(it["order"]); k=min(restrict,len(o)); idx=sorted(range(k),key=lambda j:(-s[j],j))
        out.append([o[j] for j in idx]+o[k:])
    return out
def ev(items,S,restrict=N):
    new=orders_from(items,S,restrict); rk=[rank_after(i,o) for i,o in zip(items,new)]
    g=sum(rank_after(i,o)<i["gold"] for i,o in zip(items,new)); l=sum(rank_after(i,o)>i["gold"] for i,o in zip(items,new))
    return score(rk),r1(rk),g,l
base_d=score([i["gold"] for i in dev]); base_v=score([i["gold"] for i in val])
P("기준(조건B) dev %.4f val %.4f"%(base_d,base_v))
# 현행 E 규칙(top3) 참고
bB=pick(KE,dev) if False else None
for l2 in (1e-3,1e-2,1e-1):
    w=fit_clogit(Xd,Yd,l2); Sd=Xd@w; Sv=Xv@w
    # dev는 학습에 쓰였으니 2-fold로도 확인
    idx=np.arange(len(dev)); h=len(dev)//2; cvS=np.zeros_like(Sd)
    for a,b in ((idx[:h],idx[h:]),(idx[h:],idx[:h])):
        wf=fit_clogit(Xd[a],Yd[a],l2); cvS[b]=Xd[b]@wf
    for rs in (3,5):
        r_d=ev(dev,cvS,rs); r_v=ev(val,Sv,rs)
        P("조건부로짓 l2=%g top%d | dev(CV) %.4f val %.4f R@1 %.4f | val 개선 %d 악화 %d"%(l2,rs,r_d[0],r_v[0],r_v[1],r_v[2],r_v[3]))
    if l2==1e-2: P("  가중치:"," ".join("%.2f"%x for x in w))
try:
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    def flat(X,Y):
        n,k,f=X.shape; xs=X.reshape(-1,f); ys=np.zeros(n*k); 
        for i,y in enumerate(Y):
            if y>=0: ys[i*k+y]=1
        return xs,ys
    xs,ys=flat(Xd,Yd); idx=np.arange(len(dev)); h=len(dev)//2
    def gb(Xa,Ya,Xb):
        xa,ya=flat(Xa,Ya); m=HGB(max_iter=150,learning_rate=0.05,max_depth=3,l2_regularization=1.0).fit(xa,ya)
        return m.predict_proba(Xb.reshape(-1,Xb.shape[2]))[:,1].reshape(Xb.shape[0],Xb.shape[1])
    Sv=gb(Xd,Yd,Xv); cvS=np.zeros((len(dev),N))
    for a,b in ((idx[:h],idx[h:]),(idx[h:],idx[:h])): cvS[b]=gb(Xd[a],Yd[a],Xd[b])
    for rs in (3,5):
        r_d=ev(dev,cvS,rs); r_v=ev(val,Sv,rs)
        P("GBM top%d | dev(CV) %.4f val %.4f R@1 %.4f | val 개선 %d 악화 %d"%(rs,r_d[0],r_v[0],r_v[1],r_v[2],r_v[3]))
except Exception as e: P("GBM 실패",e)
P("끝")
