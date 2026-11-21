# Kaggle 단일 스크립트: 팀 S0 + S1_noaux BLIP 점수를 제로샷 SigLIP-384 top-10 풀에 대해 계산.
# 사용: 환경변수 P1,P2,P3(네이버 대용량 링크 3개)를 설정하고 `python kaggle_s01.py`
# 산출: /kaggle/working/s01_margins.json  {qid:[{image_id,cos,S0,S1}]}  (풀 = 4000장 갤러리에서 코사인 상위 10, 코사인 순)
import json, math, os, subprocess, sys, time
from pathlib import Path

W = Path('/kaggle/working')
REPO = W / 'repo'
PINS = W / 'pins'
REL = W / 'rel'
DATA = W / 'haikuthon-train'
MODEL_DIR = REPO / 'final_ensemble'
PART4 = 'https://raw.githubusercontent.com/jaeuk6797-cell/haikuthon-mmrag/s01-assets/assets/haikuthon-88682369b8b1.part004'
SPLIT = 'https://raw.githubusercontent.com/jaeuk6797-cell/haikuthon-mmrag/s01-assets/assets/team_split.json'
PARTS = [('P1', 'haikuthon-8243a71ee07a.part001'), ('P2', 'haikuthon-80c3c6d20946.part002'), ('P3', 'haikuthon-48e9a3bea24b.part003')]
OUT = W / 's01_margins.json'


def sh(cmd, check=True):
    print('$', cmd[:150], flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode:
        raise SystemExit('실패: ' + cmd[:100])


def stage1():
    t0 = time.time()
    W.mkdir(exist_ok=True)
    # 1) 팀 요구 버전 고정 설치 (캐글 기본 환경은 건드리지 않고 별도 폴더에 설치, 단계 2는 새 프로세스)
    sh(f'pip install -q --no-deps --target {PINS} transformers==4.51.3 tokenizers==0.21.4 huggingface-hub==0.36.2')
    # 2) 저장소 final 브랜치
    if not REPO.exists():
        sh(f'git clone -q --depth 1 -b final https://github.com/jaeuk6797-cell/haikuthon-mmrag {REPO}')
    REL.mkdir(exist_ok=True)
    # 3) 가중치 조각 (네이버: 점검 오류 응답은 198바이트 -> 재시도)
    for env, name in PARTS:
        dst = REL / name
        for k in range(8):
            if dst.exists() and dst.stat().st_size == 939524096:
                break
            sh(f"wget -q -O {dst} '{os.environ[env]}'", check=False)
            if not (dst.exists() and dst.stat().st_size == 939524096):
                print('재시도', name, dst.stat().st_size if dst.exists() else -1, flush=True)
                time.sleep(10)
        assert dst.stat().st_size == 939524096, name
    sh(f'wget -q -O {REL}/haikuthon-88682369b8b1.part004 {PART4}')
    sh(f'wget -q -O {W}/team_split.json {SPLIT}')
    assert (REL / 'haikuthon-88682369b8b1.part004').stat().st_size == 2105916
    print('다운로드 완료 %.0f초' % (time.time() - t0), flush=True)
    # 4) 복원(해시 검증 포함)
    if not (MODEL_DIR / 'models').exists():
        sh(f'cd {MODEL_DIR} && python restore_model.py --assets {REL}')
    print('복원 완료 %.0f초' % (time.time() - t0), flush=True)
    # 5) 데이터
    if not (DATA / 'prompts.jsonl').exists():
        sh("pip install -q huggingface_hub", check=False)
        from huggingface_hub import snapshot_download
        snapshot_download('KimWooJin/26-haikuthon-train', repo_type='dataset', local_dir=str(DATA),
                          allow_patterns=['prompts.jsonl', 'images/*'], max_workers=8)
    print('이미지', len(os.listdir(DATA / 'images')), '%.0f초' % (time.time() - t0), flush=True)
    # 6) 단계 2: 고정 버전 transformers로 새 프로세스 실행
    env = dict(os.environ, PYTHONPATH=f'{PINS}:{MODEL_DIR}')
    r = subprocess.run([sys.executable, __file__, 'stage2'], env=env)
    if r.returncode:
        raise SystemExit('stage2 실패')
    print('전체 %.0f초' % (time.time() - t0))


def stage2():
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    print('transformers', transformers.__version__, flush=True)
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    rows = {}
    for line in open(DATA / 'prompts.jsonl', encoding='utf-8'):
        if line.strip():
            r = json.loads(line)
            rows[r['id']] = r['prompts']
    ids = sorted(rows, key=int)
    d = json.load(open(W / 'team_split.json'))
    val = d['val_ids']
    print('질의', len(val), '갤러리', len(ids), flush=True)
    dev = 'cuda'
    # 제로샷 SigLIP-384 코사인 상위 10 (우리 E 후보와 같은 K=10)
    t0 = time.time()
    SID = 'google/siglip-base-patch16-384'
    proc = AutoProcessor.from_pretrained(SID)
    net = AutoModel.from_pretrained(SID, torch_dtype=torch.float16).to(dev).eval()
    f_ = lambda o: o.pooler_output if hasattr(o, 'pooler_output') else o
    from concurrent.futures import ThreadPoolExecutor

    def load(i):
        with Image.open(DATA / 'images' / f'{i}.png') as im:
            return proc(images=im.convert('RGB'), return_tensors='pt').pixel_values[0].half()
    with ThreadPoolExecutor(4) as ex:
        px = torch.stack(list(ex.map(load, ids)))
    img = []
    with torch.inference_mode():
        for s in range(0, len(ids), 64):
            img.append(F.normalize(f_(net.get_image_features(pixel_values=px[s:s + 64].to(dev))).float(), dim=-1))
        img = torch.cat(img)
        txt = []
        for s in range(0, len(val), 128):
            tk = proc(text=[rows[i] for i in val[s:s + 128]], padding='max_length', max_length=64, truncation=True, return_tensors='pt').to(dev)
            txt.append(F.normalize(f_(net.get_text_features(**tk)).float(), dim=-1))
        txt = torch.cat(txt)
        S = (txt @ img.T).cpu()
    top = S.topk(10, dim=1)
    pool = {q: [ids[j] for j in top.indices[n].tolist()] for n, q in enumerate(val)}
    cos = {q: top.values[n].tolist() for n, q in enumerate(val)}
    del net, px, img, txt
    torch.cuda.empty_cache()
    print('SigLIP-384 풀 생성 %.0f초' % (time.time() - t0), flush=True)

    def pt(r):
        return 0.0 if r > 5 else 0.5 * ((r == 1) + 1 / math.log2(r + 1))

    def score(order):  # {q: [ids ranked]}
        return float(np.mean([pt(o.index(q) + 1 if q in o else 99) for q, o in order.items()]))
    print('풀 top10 안에 정답 포함 %.3f | 코사인 순 점수 %.4f' % (np.mean([q in pool[q] for q in val]), score({q: pool[q] for q in val})), flush=True)

    # 팀 모델
    from model import load_model, prepare_gallery, score_candidates
    t1 = time.time()
    model = load_model(str(MODEL_DIR), dev)
    queries = [{'prompt_id': q, 'prompt': rows[q]} for q in val]
    cands = [{'image_id': i, 'image_path': str(DATA / 'images' / f'{i}.png')} for i in ids]
    gal = prepare_gallery(model, cands)
    print('팀 모델 로드+갤러리 %.0f초' % (time.time() - t1), flush=True)
    t2 = time.time()
    res = score_candidates(model, queries, gal, candidate_lists=pool)
    print('S0/S1 채점 %.0f초 (질의 %d, 후보 10)' % (time.time() - t2, len(val)), flush=True)
    arms = list(next(iter(res.values()))[0]['margins'])
    print('arms', arms)
    out = {}
    for q in val:
        byid = {r['image_id']: r for r in res[q]}
        out[q] = [{'image_id': i, 'cos': c, **{a: byid[i]['margins'][a] for a in arms}} for i, c in zip(pool[q], cos[q])]
    json.dump(out, open(OUT, 'w'))
    print('저장', OUT, OUT.stat().st_size, '바이트')
    # 점수
    def zs(v):
        v = np.array(v, dtype=np.float64)
        return (v - v.mean()) / max(v.std(), 1e-8)
    for name, fn in [('코사인 순', lambda q: [x['cos'] for x in out[q]])] + \
                    [(a + ' 단독', (lambda a: lambda q: [x[a] for x in out[q]])(a)) for a in arms] + \
                    [('S0+S1 z평균', lambda q: list(sum(zs([x[a] for x in out[q]]) for a in arms)))]:
        order = {q: [out[q][k]['image_id'] for k in np.argsort(-np.array(fn(q)), kind='stable')] for q in val}
        print('  %-14s 점수 %.4f R@1 %.4f' % (name, score(order), np.mean([order[q][0] == q for q in val])))
    for w in (0.25, 0.5, 1, 2, 4):  # 코사인 z + w * (S0+S1 z평균)
        order = {}
        for q in val:
            s = zs([x['cos'] for x in out[q]]) + w * sum(zs([x[a] for x in out[q]]) for a in arms) / len(arms)
            order[q] = [out[q][k]['image_id'] for k in np.argsort(-s, kind='stable')]
        print('  cos_z + %.2f*BLIPz  점수 %.4f R@1 %.4f' % (w, score(order), np.mean([order[q][0] == q for q in val])))


if __name__ == '__main__':
    stage2() if len(sys.argv) > 1 and sys.argv[1] == 'stage2' else stage1()
