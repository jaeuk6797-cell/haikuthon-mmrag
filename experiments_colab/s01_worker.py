"""팀 S0/S1 채점 워커 (별도 프로세스, transformers 4.51.3 고정 환경).

부모 프로세스가 PYTHONPATH=<pins>:<team 폴더>로 실행한다. 표준입력으로 JSON 한 줄 명령을 받고,
표준출력으로 JSON 한 줄 이벤트를 낸다. 큰 데이터는 파일로 주고받는다.
  op=gallery {file}            : [{image_id, image_path}, ...] 로 갤러리 준비
  op=score   {file, out, chunk}: {queries:[{prompt_id,prompt}], pool:[[prompt_id,[image_id,..]],..]} 를 chunk 단위로 채점,
                                 결과 파일 <out>.<i>.json = [[prompt_id, [[image_id, {arm: margin}], ..]], ..]
  op=quit
"""
import json
import sys
import time
import traceback


def emit(**kw):
    print(json.dumps(kw), flush=True)


def main():
    import torch
    team = sys.argv[1]
    t0 = time.time()
    from model import load_model, prepare_gallery, score_candidates  # 팀 model.py (PYTHONPATH로 지정)
    m = load_model(team, torch.device("cuda"))
    emit(ev="ready", sec=round(time.time() - t0, 1))
    gal = None
    for line in sys.stdin:
        if not line.strip():
            continue
        cmd = json.loads(line)
        try:
            if cmd["op"] == "quit":
                break
            if cmd["op"] == "gallery":
                t = time.time()
                items = json.load(open(cmd["file"], encoding="utf-8"))
                gal = prepare_gallery(m, items)
                emit(ev="gallery", sec=round(time.time() - t, 1), n=len(items))
            elif cmd["op"] == "score":
                t = time.time()
                job = json.load(open(cmd["file"], encoding="utf-8"))
                qs, pool = job["queries"], {k: v for k, v in job["pool"]}
                step = int(cmd.get("chunk", 400))
                for i in range(0, len(qs), step):
                    sub = qs[i:i + step]
                    res = score_candidates(m, sub, gal, candidate_lists={q["prompt_id"]: pool[q["prompt_id"]] for q in sub})
                    rows = [[q["prompt_id"], [[r["image_id"], {a: float(v) for a, v in r["margins"].items()}] for r in res[q["prompt_id"]]]]
                            for q in sub]
                    fn = "%s.%d.json" % (cmd["out"], i)
                    json.dump(rows, open(fn, "w", encoding="utf-8"))
                    emit(ev="chunk", file=fn, i=i, sec=round(time.time() - t, 1))
                emit(ev="done", sec=round(time.time() - t, 1))
        except Exception:
            emit(ev="error", msg=traceback.format_exc()[-1200:])


if __name__ == "__main__":
    main()
