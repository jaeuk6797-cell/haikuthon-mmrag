# HAIKUTHON 이미지 검색 · 팀 앙상블 인계

팀원이 학습 없이 같은 후보 이미지에 우리 모델과 자신의 모델로 각각 점수를 매기고, 두 점수를 합쳐 Top-5를 만드는 패키지입니다. 대회 서버에서 사용하는 기본 진입점은 `model.py`의 `load_model()`과 `predict()`입니다.

## GitHub에 올리는 사람

1. 이 프로젝트에서는 기존 Git 저장소의 `final_ensemble/`에 소스를 준비했습니다. 해당 디렉터리를 커밋합니다. 별도 저장소를 사용할 경우 `github_repo/`의 내용을 넣어도 됩니다. `.gitattributes`와 `weights_manifest.json`도 포함합니다.
2. 같은 커밋을 가리키는 GitHub Release를 만들고, `release_assets/`의 **모든 `.partNNN` 파일**을 원래 이름으로 첨부합니다. 각 파일은 큰 추론 ZIP을 분할한 조각입니다.
3. 팀원에게 해당 릴리스 링크를 전달합니다. 소스와 가중치 조각 모두 같은 릴리스에서 받아야 합니다. 대회에 사용하는 파일은 전체 `HAIKUTHON_Final_Ensemble.zip`입니다.

## 팀원이 받았을 때

저장소를 같은 릴리스의 태그로 체크아웃하고 `model.py`가 있는 디렉터리로 들어갑니다. 기존 팀 저장소에서는 `cd final_ensemble`입니다. 이 위치에 `release_assets/`를 만들고 릴리스의 조각 네 개를 넣습니다.

```bash
python restore_model.py --assets release_assets
python -m pip install -r requirements.txt
```

복원기는 각 조각, 합친 ZIP, 추론 코드, 개별 모델 파일의 SHA-256을 검사한 다음 `models/`를 설치합니다. 전체 ZIP을 이미 받았다면 첫 줄을 `python restore_model.py --archive /path/to/HAIKUTHON_Final_Ensemble.zip`으로 바꿀 수 있습니다. Python 3.12 및 서로 호환되는 PyTorch·torchvision이 설치된 환경에서 실행합니다. 대회 환경에서는 기본 제공 PyTorch·torchvision·packaging을 유지합니다. 모델, 토크나이저, 전처리 설정은 모두 로컬 파일에서 로드합니다.

## 앙상블에서 쓸 점수

두 모델이 **같은 질의에 대해 같은 후보 이미지 ID**를 채점하게 합니다. 팀원의 검색 Top-8, 우리 검색 Top-8 또는 두 목록의 합집합 중 하나를 개발 데이터에서 정합니다. 양쪽이 이미 서로 다른 후보를 골랐다면 합집합의 모든 후보를 양쪽으로 다시 채점합니다. 이미지 ID는 앞자리 0을 보존한 문자열로 전달합니다.

```python
from model import load_model, prepare_gallery, retrieve_candidates, score_candidates
from ensemble_scores import combine

model = load_model(".", "cuda")
gallery = prepare_gallery(model, candidates)
pool = retrieve_candidates(model, queries, gallery, top_k=8)
# 팀원이 이미 공통 후보를 골랐다면 위 pool을 그 딕셔너리로 대체합니다.
ours = score_candidates(model, queries, gallery, candidate_lists=pool)
# teammate_rows에도 각 질의의 같은 후보 전체에 대해 image_id와 score가 있어야 합니다.
predictions = combine(ours, teammate_rows, our_weight=0.5, top_k=5)
```

`queries`는 `[{"prompt_id":"q_001","prompt":"A cat on a sofa."}]`, `candidates`는 `[{"image_id":"img_001","image_path":"/data/img_001.png"}, ...]` 형식입니다. `pool`은 질의 ID를 이미지 ID 목록에 연결합니다. 두 점수 파일은 `{"q_001":[{"image_id":"img_001","score":1.7}, ...]}` 형식이고, 큰 점수가 더 좋은 이미지입니다. 예시에 있는 생략 기호는 실제 JSON에서 제외합니다.

```bash
python ensemble_scores.py score --queries queries.json --candidates candidates.json --pool pool.json --device cuda --out ours.json
python ensemble_scores.py combine --ours ours.json --teammate theirs.json --our-weight 0.5 --out top5.json
```

`ours.json`에는 후보별 SigLIP 코사인 유사도(`siglip_cosine`), 각 BLIP의 매칭·비매칭 로짓 차이(`margins`), 결합 점수(`score`), 순위(`rank`)가 들어갑니다. 단독 BLIP의 점수는 원시 로짓 차이입니다. 복수 BLIP는 **한 질의의 후보 안에서** 각 모델의 점수를 표준화한 뒤 가중평균합니다. 팀 모델과의 결합도 질의별로 같은 계산을 합니다. `0.5`는 팀원 조합의 출발값입니다. 최종 비중은 운영진 학습 데이터에서 분리한 검증셋에서 정합니다. 서버의 CSV는 사이트에서 받은 그대로 Kaggle에 제출합니다.

## 기본 모델과 선택 구성

| 구성 | 실행 방법 | 적용 근거 |
|---|---|---|
| 기본 S0 + S1_noaux | `load_model(".", "cuda")` | 형제 후보 8장 중 정답을 고르는 BLIP 두 개를 질의별 표준화 후 1대1로 결합합니다. S1_noaux에는 학습 시 적용한 정답 보존 증강이 반영돼 있습니다. |
| 단독 S0 또는 S1_noaux | `arms=["S0"]` 또는 `arms=["S1_noaux"]` | 공유 이미지 특징에 텍스트 인코더 한 개만 적용합니다. 팀원 모델과 짝을 지어 가중치별 기여나 시간을 비교할 수 있습니다. |
| 공개 BLIP A + Hard Negative C | `arms=["A","C"]` | 과거 ABC 분할에서 A의 점수 0.51582가 1대1 결합 시 0.52244였습니다. 이 릴리스는 현재 형제 분할의 SigLIP 헤드를 사용합니다. C의 과거 학습 데이터가 현재 검증셋 일부와 겹쳐 그 검증셋으로 C의 개선량을 추정할 수 없습니다. |
| Top-2 이미지 비교기 | `load_model(".", "cuda", arms=["A"], use_pair=True)` | 공개 A에서 800건 중 Top-1을 30건 고치고 23건 잃었습니다. 개발셋에서 고정한 0.75 기준을 적용합니다. 비교기 실행 시 최종 순서는 `rank`이고 `score`는 교환 전 값입니다. |
| Top-10 후보 재정렬 | `score_candidates(model, queries, gallery, width=10)` | 같은 검색기로 10장을 가져와 점수를 매깁니다. 최종 반환은 상위 5장입니다. |
| 초기 공간·색상 대조와 순위 결합 | `legacy_contrast.py`의 `legacy_scores()` | 원본 SigLIP Top-5에서 좌우 반전, 색상 교환, 두 점수의 조합, 검색기와 공개 BLIP의 순위 결합을 선택할 수 있습니다. 별도 검출기는 사용하지 않습니다. 사용 예는 모듈 첫 부분에 있습니다. |

기본 추론은 PNG를 한 번 디코딩해 SigLIP 224px와 BLIP 384px 입력을 만들고, 두 BLIP가 이미지 인코더 출력을 공유합니다. 검출기 기반 ROI는 서버 시간 초과로 제외했습니다. 색·방향 오답 보조 손실을 더한 S1은 같은 800건에서 0.75614였고, 해당 손실을 뺀 S1_noaux는 0.76119였습니다. 점수가 더 높았던 S1_noaux 가중치를 배포합니다.

## 검증 결과를 읽는 법

원본 S0의 Top-8 후보로 맞춘 **내부 검증 800건**에서 S0는 정답 1위 527건·대회식 점수 0.75637, S1_noaux는 534건·0.76119, 기본 결합은 **541건·0.76913**이었습니다. 정답 Top-5는 각각 794·792·795건입니다. 후보가 같았던 797건은 저장된 점수를 이미지 ID로 맞췄고, 후보가 달랐던 3건은 운영진 PNG와 두 실제 가중치로 CPU에서 다시 채점했습니다. 같은 검증 결과를 보고 선택한 조합입니다. [세부 수치](validation_supplement.json)

이 구성의 공식 8,000질의·16,000이미지 추론 시간과 Public 점수는 미측정입니다. 과거 A 단독의 시간은 이번 두 BLIP 조합의 시간이 아닙니다. 패키지 내부 점검과 오프라인 소규모 실추론은 아래 명령으로 재현할 수 있습니다. 프로그램은 공식 학습 이미지와 프롬프트를 로컬에서 읽습니다.

```bash
python check_package.py HAIKUTHON_Final_Ensemble.zip --image-dir data/train/images --prompts data/train/prompts.jsonl --report qa.json
```

사전학습 모델 출처와 라이선스는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 있습니다.
