# 캐글: E 구성(제로샷 SigLIP-384 + 조건B + 색 + OWL)에 팀 S0/S1 점수(s01_margins.json)를 융합해 검증 800에서 평가한다.
# 선행: /kaggle/working/s01_margins.json, /kaggle/working/haikuthon-train, /kaggle/working/team_split.json (kaggle_s01.py가 만든 것)
# 실행: python -u kaggle_fuse.py   (기본 캐글 환경, 고정 버전 설치 없음)
import gc, json, math, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

W = Path('/kaggle/working')
S2 = W / 'sub2'
REPO = W / 'repo2'
DATA = W / 'haikuthon-train'
BR = os.environ.get('FUSE_BRANCH', 'jaeuk')


def sh(c):
    print('$', c[:120], flush=True)
    subprocess.run(c, shell=True, check=True)


if not REPO.exists():
    sh(f'git clone -q --depth 1 -b {BR} https://github.com/jaeuk6797-cell/haikuthon-mmrag {REPO}')
if S2.exists():
    shutil.rmtree(S2)
S2.mkdir()
shutil.copytree(REPO / 'submission' / 'final_ft' / 'haiku', S2 / 'haiku')
shutil.copy(REPO / 'experiments_colab' / 'model_s0.py', S2 / 'model.py')
M = S2 / 'models'
M.mkdir()
from transformers import AutoModel, AutoProcessor, OwlViTForObjectDetection, OwlViTProcessor
SID = 'google/siglip-base-patch16-384'
n = AutoModel.from_pretrained(SID, dtype=torch.float16)
n.save_pretrained(M / 'siglip')
AutoProcessor.from_pretrained(SID).save_pretrained(M / 'siglip')
del n
os.symlink(M / 'siglip', M / 'siglip_zs')
OID = 'google/owlvit-base-patch32'
o = OwlViTForObjectDetection.from_pretrained(OID, dtype=torch.float16)
o.save_pretrained(M / 'owl')
OwlViTProcessor.from_pretrained(OID).save_pretrained(M / 'owl')
del o
gc.collect()
json.dump({'K': 10, 'method': 'B', 'lam': 3.2, 'itm': {'use': False}, 'owl': {'use': True}, 'color': {'use': True}}, open(M / 'haiku_config.json', 'w'))

sys.path.insert(0, str(S2))
import importlib.util
spec = importlib.util.spec_from_file_location('submodel2', S2 / 'model.py')
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)
from haiku.data import load_rows, to_queries, to_candidates

rows = load_rows(str(DATA))
by = {r['id']: r for r in rows}
d = json.load(open(W / 'team_split.json'))
vrows = [by[i] for i in d['val_ids']]
q, cand = to_queries(vrows), to_candidates(rows)
cid = [c['image_id'] for c in cand]
marg = json.load(open(W / 's01_margins.json'))
prompt2q = {by[i]['prompt']: i for i in d['val_ids']}
tab = {}
for qid, lst in marg.items():
    for x in lst:
        tab[(qid, x['image_id'])] = (x['S0'], x['S1_noaux'])
print('margins', len(marg), 'pairs', len(tab), flush=True)

# 이미지 인코딩 캐시 (설정마다 다시 인코딩하지 않도록)
_enc = sm.encode_images
_cache = {}


def enc_cached(model, paths):
    k = (len(paths), paths[0], paths[-1])
    if k not in _cache:
        _cache[k] = _enc(model, paths)
    return _cache[k]


sm.encode_images = enc_cached
sm.encode_images_raw = lambda model, paths: (enc_cached(model, paths), True)

W0, W1 = 1.0, 1.0   # S0, S1 z 가중
MODE = {'on': False}


def itm_stub(model, raw, img_idx, texts):
    x = np.zeros((len(texts), 2), dtype=np.float64)
    for i, (j, t) in enumerate(zip(img_idx, texts)):
        v = tab.get((prompt2q[t], cid[j]))
        x[i] = v if v is not None else (np.nan, np.nan)
    kk = MODE['k']
    x = x.reshape(-1, kk, 2)
    out = np.zeros((x.shape[0], kk))
    for r in range(x.shape[0]):
        g = x[r]
        if np.isnan(g).any():
            continue
        z = (g - g.mean(0)) / np.maximum(g.std(0), 1e-6)
        out[r] = (W0 * z[:, 0] + W1 * z[:, 1]) / (W0 + W1)
    return torch.tensor(out.reshape(-1), dtype=torch.float32, device='cuda')


sm.itm_scores = itm_stub
model = sm.load_model(S2, 'cuda')
model['blip'] = object()
model['blip_proc'] = None


def score(res):
    pt = []
    for qq, lst in res.items():
        r = lst.index(qq) + 1 if qq in lst else 99
        pt.append(0.0 if r > 5 else 0.5 * ((r == 1) + 1 / math.log2(r + 1)))
    return float(np.mean(pt)), float(np.mean([lst[0] == qq for qq, lst in res.items()]))


base = {"K": 10, "method": "B", "lam": 3.2,
        "kinds": ["color", "pattern", "material", "shape", "state", "pose", "count", "color_binding", "count_binding", "relation_left_right", "relation_above_below", "relation_on_inside"],
        "itm": {"use": True, "k": 10, "lam": 0.1, "tau": None, "skip_after_sec": 900},
        "owl": {"use": True, "thr": 0.1, "margin": 0.03, "start_before_sec": 600, "stop_after_sec": 740, "kinds": ["left_of", "above", "on", "inside", "rides", "next_to"], "top_n": 3},
        "color": {"use": True, "mu": 2.0, "thr": 0.1, "pad": 0.1, "start_before_sec": 680}}


def cfg(lam, owl=True, color=True, itm=True, w=(1.0, 1.0)):
    c = json.loads(json.dumps(base))
    c['itm'].update({'use': itm, 'lam': lam})
    c['owl']['use'] = owl
    c['color']['use'] = color
    W0_, W1_ = w
    global W0, W1
    W0, W1 = W0_, W1_
    return c


MODE['k'] = 10
runs = [('E (S 없음)', cfg(0, itm=False))] + \
       [('E + S z-융합 lam=%g' % l, cfg(l)) for l in (0.2, 1.0, 5.0)] + \
       [('S만(조건·색·OWL 끔) lam=5', cfg(5.0, owl=False, color=False))]
res_all = []
for name, c in runs:
    model['cfg'] = c
    t = time.time()
    r = sm.predict(model, q, cand)
    s, r1 = score(r)
    res_all.append((name, s))
    print('[%s] 점수 %.4f R@1 %.4f | %.0f초 | %s' % (name, s, r1, time.time() - t, {k: v for k, v in model['last_timing'].items() if k in ('total_sec', 'color_swapped', 'owl_swapped')}), flush=True)
print('요약', res_all)
print('끝')
