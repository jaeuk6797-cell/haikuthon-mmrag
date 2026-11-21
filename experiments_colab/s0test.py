# 코랩: FT(3200 학습, 정직한 검증) 검색 + 팀원 S0 BLIP 재정렬 조합 검증. 선행: /content/ft_3200_fp16.safetensors, /content/s0_weights.safetensors, /content/sub, /content/model_s0.py
import os, sys, json, shutil, time, math, gc
from pathlib import Path
import numpy as np, torch
for k in ('net','PX','best','owl','cc','F_img','F_txt','Z_img','Z_txt','model','sm'): globals().pop(k, None)
gc.collect(); torch.cuda.empty_cache()
SUB, S2 = Path('/content/sub'), Path('/content/sub2')
if S2.exists(): shutil.rmtree(S2)
S2.mkdir()
for n in ('haiku', 'requirements.txt'): (shutil.copytree if (SUB / n).is_dir() else shutil.copy)(SUB / n, S2 / n)
shutil.copy('/content/model_s0.py', S2 / 'model.py')
M2 = S2 / 'models'; M2.mkdir()
shutil.copy(SUB / 'models' / 'haiku_config.json', M2 / 'haiku_config.json')
os.symlink(SUB / 'models' / 'owl', M2 / 'owl'); os.symlink(SUB / 'models' / 'siglip_zs', M2 / 'siglip_zs')
shutil.copytree(SUB / 'models' / 'siglip_zs', M2 / 'siglip')
shutil.copy('/content/ft_3200_fp16.safetensors', M2 / 'siglip' / 'model.safetensors')
from transformers import BlipForImageTextRetrieval, BlipProcessor
BL = 'Salesforce/blip-itm-base-coco'
b = BlipForImageTextRetrieval.from_pretrained(BL, dtype=torch.float16); b.save_pretrained(M2 / 'blip', safe_serialization=True); BlipProcessor.from_pretrained(BL).save_pretrained(M2 / 'blip'); del b
shutil.copy('/content/s0_weights.safetensors', M2 / 'blip' / 's0_weights.safetensors')
for k in [k for k in sys.modules if k.startswith('haiku') or k == 'submodel2']: sys.modules.pop(k)
sys.path.insert(0, str(S2))
import importlib.util
spec = importlib.util.spec_from_file_location('submodel2', S2 / 'model.py'); sm = importlib.util.module_from_spec(spec); spec.loader.exec_module(sm)
from haiku.data import load_rows, to_queries, to_candidates
rows = load_rows('/content/haikuthon-train'); by = {r['id']: r for r in rows}
d = json.load(open('/content/team_split.json')); vrows = [by[i] for i in d['val_ids']]
def score(res):
    pt = []
    for q, lst in res.items():
        r = lst.index(q) + 1 if q in lst else 99
        pt.append(0.0 if r > 5 else 0.5 * ((r == 1) + 1 / math.log2(r + 1)))
    return float(np.mean(pt))
base = json.loads(open(M2 / 'haiku_config.json').read())
base['itm'] = {'use': True, 'k': 8, 'lam': 0.0, 'tau': None, 'skip_after_sec': 900}
def cfg(**kw):
    c = json.loads(json.dumps(base))
    c['itm'].update(kw.pop('itm', {}))
    for k, v in kw.items():
        if isinstance(v, dict): c[k].update(v)
        else: c[k] = v
    return c
model = sm.load_model(S2, 'cuda'); print('S0 키 적용 수:', model.get('s0_keys'), '| blip', model['blip'] is not None, flush=True)
q, cand = to_queries(vrows), to_candidates(rows)
runs = [
 ('FT 파이프라인(ITM 없음)', cfg(itm={'lam': 0.0})),
 ('FT + S0 ITM(k8) lam=1.0 (ITM 지배, 색/OWL 끔)', cfg(itm={'lam': 1.0}, owl={'use': False}, color={'use': False})),
 ('FT + S0 ITM(k8) lam=0.1 + 색/OWL', cfg(itm={'lam': 0.1})),
 ('FT + S0 ITM(k8) lam=0.03 + 색/OWL', cfg(itm={'lam': 0.03})),
 ('FT + S0 ITM(k8) lam=1.0 + 색/OWL', cfg(itm={'lam': 1.0})),
]
for name, c in runs:
    model['cfg'] = c; t = time.time(); res = sm.predict(model, q, cand)
    print('[%s] 점수 %.4f | %.0f초 | itm_end %s total %s' % (name, score(res), time.time() - t, model['last_timing'].get('itm_end_sec'), model['last_timing'].get('total_sec')), flush=True)
print('끝')
