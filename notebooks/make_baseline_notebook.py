"""baseline_kaggle.ipynb 생성 스크립트 (python make_baseline_notebook.py)."""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# HAIKUTHON 기준선 실행 (Kaggle / Colab)

스타터킷 `train.py`를 그대로 돌려 기준선 점수를 확인하고, 유형별 오답 분석과 추론 시간 추정까지 한 번에 한다.

**Kaggle 설정**: 우측 Settings에서 Accelerator = GPU (T4 x2 또는 P100), Internet = On.
Colab에서는 런타임 유형을 T4 GPU로 바꾼다. 위에서부터 순서대로 실행하면 된다.
""")

code("""
import json, math, os, re, subprocess, sys, time, zipfile
from pathlib import Path

WORK = Path("/kaggle/temp") if Path("/kaggle").exists() else Path("/content/work")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
print("작업 폴더:", WORK)
""")

code("""
!pip install -q "transformers==5.22.2" sentencepiece protobuf safetensors "huggingface_hub>=1.0"
""")

md("## 1. 데이터와 스타터킷 받기")

code("""
from huggingface_hub import snapshot_download

DATA = WORK / "haikuthon-train"
snapshot_download(
    repo_id="KimWooJin/26-haikuthon-train",
    repo_type="dataset",
    allow_patterns=["prompts.jsonl", "images/*.png", "starterkit.zip"],
    local_dir=DATA,
)
with zipfile.ZipFile(DATA / "starterkit.zip") as z:
    for name in z.namelist():
        if not name.endswith(".zip"):  # 예제 제출 ZIP(750MB)은 풀지 않는다
            z.extract(name, WORK)
KIT = WORK / "starterkit"
print(sorted(p.name for p in KIT.iterdir()))
print("이미지 수:", len(list((DATA / "images").glob("*.png"))))
""")

md("## 2. 스타터킷 기준선 학습 (기본 설정: 1에포크, 배치 16, 출력 헤드만 학습)")

code("""
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음")
t0 = time.time()
!cd {KIT} && python train.py --data-dir {DATA}
print(f"학습+검증 소요: {(time.time() - t0) / 60:.1f}분")
""")

md("""
스타터킷이 보고한 값은 Recall@1 0.6925 / NDCG@5 0.8752 / 평균 0.7839 (검증 풀 800장).
위 출력이 이와 비슷하면 환경이 정상이다.
""")

md("## 3. 유형별 오답 분석 (검증 800쌍)")

code("""
sys.path.insert(0, str(KIT))
from train import load_and_split
from sample_submission.model import load_model, predict, read_image

train_rows, valid_rows = load_and_split(DATA, 42)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(KIT / "sample_submission", device)

queries = [{"prompt_id": r["id"], "prompt": r["prompts"]} for r in valid_rows]
candidates = [{"image_id": r["id"], "image_path": r["image_path"]} for r in valid_rows]
preds = predict(model, queries, candidates, top_k=5)


def score(rank):
    if rank is None:
        return 0.0
    return 0.5 * ((rank == 1) + 1 / math.log2(rank + 1))


records = []
for r in valid_rows:
    top5 = preds[r["id"]]
    rank = top5.index(r["id"]) + 1 if r["id"] in top5 else None
    records.append({"id": r["id"], "prompt": r["prompts"], "rank": rank, "score": score(rank), "top1": top5[0]})

overall = sum(x["score"] for x in records) / len(records)
r1 = sum(x["rank"] == 1 for x in records) / len(records)
print(f"검증 평균 {overall:.4f} | Recall@1 {r1:.4f}")

groups = {
    "좌/우 관계": lambda t: bool(re.search(r"to the (left|right) of", t.lower())),
    "위/아래 관계": lambda t: bool(re.search(r"\\b(above|below)\\b", t.lower())),
    "next to": lambda t: "next to" in t.lower(),
    "on / under": lambda t: bool(re.search(r"\\b(resting on|rests on|under|beneath)\\b", t.lower())),
    "개수 2 이상(two/three)": lambda t: bool(re.search(r"\\b(two|three)\\b", t.lower())),
    "무늬(striped 등)": lambda t: bool(re.search(r"(striped|polka-dotted|checkered|solid-colored)", t.lower())),
    "색 단어 3개 이상": lambda t: len(re.findall(r"\\b(red|orange|yellow|green|blue|purple|pink|brown|black|white|gray|beige)\\b", t.lower())) >= 3,
    "객체 4개 이상('One' 4회+)": lambda t: len(re.findall(r"\\bone\\b", t.lower())) >= 4,
    "프롬프트 길이 상위 25%": None,
}
lens = sorted(len(x["prompt"].split()) for x in records)
q75 = lens[int(len(lens) * 0.75)]
groups["프롬프트 길이 상위 25%"] = lambda t: len(t.split()) >= q75

print(f"{'그룹':<26}{'N':>5}{'평균점수':>10}{'Recall@1':>10}")
for name, fn in groups.items():
    sel = [x for x in records if fn(x["prompt"])]
    if sel:
        print(f"{name:<26}{len(sel):>5}{sum(x['score'] for x in sel) / len(sel):>10.3f}"
              f"{sum(x['rank'] == 1 for x in sel) / len(sel):>10.3f}")
""")

md("""
## 4. 평가 규모(쿼리 8,000개 x 이미지 16,000장) 추론 시간 추정

학습 이미지 4,000장을 4번 복제해 16,000장짜리 후보 풀을, 프롬프트를 2번 복제해 8,000개 쿼리를 만들어 `predict` 시간을 잰다.
평가 서버는 L4이고 여기는 T4/P100이라 절대 시간은 다르다. **이미지 디코딩(CPU) 비중**이 얼마인지가 핵심이다.
""")

code("""
all_rows = train_rows + valid_rows
big_candidates = [
    {"image_id": f"{r['id']}_{k}", "image_path": r["image_path"]}
    for k in range(4) for r in all_rows
]
big_queries = [
    {"prompt_id": f"{r['id']}_{k}", "prompt": r["prompts"]}
    for k in range(2) for r in all_rows
]
print(len(big_candidates), "이미지 /", len(big_queries), "쿼리")

# (a) 이미지 디코딩 + 전처리만 (CPU)
sample = big_candidates[:300]
t0 = time.time()
for item in sample:
    read_image(item["image_path"])
decode_ms = (time.time() - t0) / len(sample) * 1000
print(f"PNG 디코딩 평균 {decode_ms:.1f}ms/장 -> 16,000장 약 {decode_ms * 16000 / 1000:.0f}초 (단일 프로세스)")

# (b) 전체 predict
torch.cuda.synchronize() if torch.cuda.is_available() else None
t0 = time.time()
_ = predict(model, big_queries, big_candidates, top_k=5)
torch.cuda.synchronize() if torch.cuda.is_available() else None
total = time.time() - t0
print(f"predict 전체: {total:.0f}초 ({total / 60:.1f}분), 제한 900초 대비 {total / 900 * 100:.0f}%")
print("CPU 코어 수:", os.cpu_count())
""")

md("## 5. 프롬프트 토큰 길이 (SigLIP 텍스트 최대 64토큰)")

code("""
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(KIT / "sample_submission" / "models", local_files_only=True)
all_prompts = [r["prompts"] for r in train_rows + valid_rows]
lens = [len(ids) for ids in processor.tokenizer(all_prompts, padding=False, truncation=False)["input_ids"]]
lens.sort()
n = len(lens)
print("min/median/p99/max:", lens[0], lens[n // 2], lens[int(n * 0.99)], lens[-1])
print("64 초과:", sum(l > 64 for l in lens), f"({100 * sum(l > 64 for l in lens) / n:.1f}%)")
""")

md("## 6. 결과 저장")

code("""
result = {
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "valid_score": overall,
    "valid_recall1": r1,
    "png_decode_ms": decode_ms,
    "predict_seconds_16k_8k": total,
}
out = Path("/kaggle/working") if Path("/kaggle/working").exists() else WORK
(out / "baseline_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out_path = Path(__file__).with_name("baseline_kaggle.ipynb")
out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("작성:", out_path)
