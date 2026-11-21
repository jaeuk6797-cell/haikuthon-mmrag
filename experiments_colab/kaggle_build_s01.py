# 캐글: E 패키지(sub2) + 팀 S0/S1(repo/final_ensemble, pins) 융합 제출 패키지(pkg) 조립 + 800질의/4000장 시험
import json, math, os, shutil, subprocess, sys, time
from pathlib import Path
W = Path('/kaggle/working'); S2 = W / 'sub2'; PKG = W / 'pkg'; PINS = W / 'pins'; TEAM = W / 'repo' / 'final_ensemble'
DATA = W / 'haikuthon-train'; HERE = Path(__file__).resolve().parent
def sh(c): print('$', c[:140], flush=True); subprocess.run(c, shell=True, check=True)
if PKG.exists(): shutil.rmtree(PKG)
sh(f'cp -al {S2} {PKG}')
sh(f'rm -rf {PKG}/models/blip {PKG}/s01')
os.makedirs(PKG / 's01', exist_ok=True)
sh(f'cp {HERE}/model_s01.py {PKG}/model.py; cp {HERE}/s01_worker.py {PKG}/s01/worker.py; cp {HERE}/haiku_config_s01.json {PKG}/models/haiku_config.json')
sh(f'cp -al {PINS} {PKG}/s01/pins; cp -al {TEAM} {PKG}/s01/team')
sh(f'rm -rf {PKG}/s01/team/.git {PKG}/s01/team/__pycache__; du -sh {PKG} | tail -1; ls {PKG} {PKG}/s01 {PKG}/models')
if '--build-only' in sys.argv: raise SystemExit
# 시험: 팀 val 800질의, 갤러리 4000장
sys.path.insert(0, str(PKG))
import torch, numpy as np
import model as M
rows = {}
for l in open(DATA / 'prompts.jsonl', encoding='utf-8'):
    if l.strip(): r = json.loads(l); rows[r['id']] = r['prompts']
ids = sorted(rows, key=int); val = json.load(open(W / 'team_split.json'))['val_ids']
cands = [{'image_id': i, 'image_path': str(DATA / 'images' / f'{i}.png')} for i in ids]
queries = [{'prompt_id': q, 'prompt': rows[q]} for q in val]
t = time.time(); model = M.load_model(str(PKG), 'cuda'); print('load %.0f초' % (time.time() - t), 's01 ok:', model['s01'].ok if model['s01'] else None, model['s01'].err if model['s01'] else '', flush=True)
t = time.time(); res = M.predict(model, queries, cands); tt = time.time() - t
def pt(o, q):
    r = o.index(q) + 1 if q in o else 99
    return 0.0 if r > 5 else 0.5 * ((r == 1) + 1 / math.log2(r + 1))
print('점수 %.4f R@1 %.4f | predict %.0f초 | %s' % (np.mean([pt(res[q], q) for q in val]), np.mean([res[q][0] == q for q in val]), tt, model['last_timing']), flush=True)
print('끝')
