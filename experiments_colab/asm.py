# 코랩: 제출 폴더 조립 + 회귀 테스트(A) + FT 구성 점검(B). 선행: /content/final_patch.zip, /content/ft_all4000_fp16.safetensors
import os, sys, json, shutil, zipfile, time, math, gc
from pathlib import Path
import numpy as np, torch
for k in ('net','PX','best','owl','cc','F_img','F_txt','Z_img','Z_txt'): globals().pop(k, None)
gc.collect(); torch.cuda.empty_cache()
SUB = Path('/content/sub'); W = Path(os.environ.get('V_W', '/content/ft_all4000_fp16.safetensors'))
if SUB.exists(): shutil.rmtree(SUB)
zipfile.ZipFile('/content/final_patch.zip').extractall(SUB)
M = SUB / 'models'
from transformers import AutoModel, AutoProcessor, OwlViTForObjectDetection, OwlViTProcessor
SIG, OWL = "google/siglip-base-patch16-384", "google/owlvit-base-patch32"
t0 = time.time()
m = AutoModel.from_pretrained(SIG, dtype=torch.float16); m.save_pretrained(M / 'siglip_zs', safe_serialization=True)
AutoProcessor.from_pretrained(SIG).save_pretrained(M / 'siglip_zs')
shutil.copytree(M / 'siglip_zs', M / 'siglip')
o = OwlViTForObjectDetection.from_pretrained(OWL, dtype=torch.float16); o.save_pretrained(M / 'owl', safe_serialization=True)
OwlViTProcessor.from_pretrained(OWL).save_pretrained(M / 'owl')
del m, o; gc.collect()
print('가중치 조립 %.0f초' % (time.time() - t0), flush=True)
have_ft = W.exists()
if have_ft:
    from safetensors.torch import load_file
    ft = load_file(str(W)); zs = load_file(str(M / 'siglip_zs' / 'model.safetensors'))
    assert set(ft) == set(zs), ('키 불일치', len(set(ft) ^ set(zs)))
    assert all(ft[k].shape == zs[k].shape and ft[k].dtype == zs[k].dtype for k in ft), '모양/자료형 불일치'
    shutil.copy(W, M / 'siglip' / 'model.safetensors')
    diff = sum(float((ft[k].float() - zs[k].float()).abs().sum() > 0) for k in ft)
    print('FT 가중치 교체 완료: 키 %d개, 제로샷과 다른 텐서 %d개' % (len(ft), diff), flush=True)
    del ft, zs
else:
    print('FT 가중치 파일 없음 -> models/siglip은 제로샷 그대로', flush=True)
for k in [k for k in sys.modules if k.startswith('haiku') or k == 'submodel']: sys.modules.pop(k)
sys.path.insert(0, str(SUB))
import importlib.util
spec = importlib.util.spec_from_file_location('submodel', SUB / 'model.py'); sm = importlib.util.module_from_spec(spec); spec.loader.exec_module(sm)
from haiku.data import load_rows, to_queries, to_candidates
from haiku.siblings import build_groups, group_split, parse_prompt
rows = load_rows('/content/haikuthon-train'); parsed = {r['id']: parse_prompt(r['prompt']) for r in rows}
_, members = build_groups(rows, parsed); tr, va = group_split(rows, members, 42)
by = {r['id']: r for r in rows}; vrows = [by[i] for i in va]
def score(res):
    pt = []
    for q, lst in res.items():
        r = lst.index(q) + 1 if q in lst else 99
        pt.append(0.0 if r > 5 else 0.5 * ((r == 1) + 1 / math.log2(r + 1)))
    return float(np.mean(pt))
model = sm.load_model(SUB, 'cuda')
print('모델 로드: net_zs=%s owl=%s blip=%s' % (model['net_zs'] is not None, model['owl'] is not None, model['blip'] is not None), flush=True)
E = json.loads('''{"K":10,"method":"B","lam":3.2,"kinds":["color","pattern","material","shape","state","pose","count","color_binding","count_binding","relation_left_right","relation_above_below","relation_on_inside"],"itm":{"use":false},"owl":{"use":true,"thr":0.1,"margin":0.03,"start_before_sec":600,"stop_after_sec":740,"kinds":["left_of","above","on","inside","rides","next_to"],"top_n":3},"color":{"use":true,"mu":2.0,"thr":0.1,"pad":0.1,"start_before_sec":680}}''')
FTC = json.loads((M / 'haiku_config.json').read_text())
q800, c800 = to_queries(vrows), to_candidates(vrows)
# A) 회귀: 제로샷 SigLIP + E 설정(BLIP 제외), 검증 800 풀 -> 실험값 0.6175
saved = model['net']; model['net'] = model['net_zs']; model['cfg'] = E
res = sm.predict(model, q800, c800); print('[A 회귀] 제로샷+E설정 검증800 점수 %.4f (기대 0.6175) | %s' % (score(res), model['last_timing']), flush=True)
model['net'] = saved
# B) FT 구성: 검증 800 풀 (FT가 4000쌍 전부 학습이면 누수라 정확도 참고 불가, 동작/시간 확인용)
model['cfg'] = FTC
res = sm.predict(model, q800, c800); print('[B FT구성 800풀] 점수 %.4f (FT=%s%s) | %s' % (score(res), 'all4000' if have_ft else '제로샷', ', 누수' if have_ft else '', model['last_timing']), flush=True)
# C) 시간 감각: 4000장 풀 x 검증 800 질의
t = time.time(); res = sm.predict(model, q800, to_candidates(rows)); print('[C 4000풀 x 800질의] 점수 %.4f | %.0f초 | %s' % (score(res), time.time() - t, model['last_timing']), flush=True)
print('끝')
