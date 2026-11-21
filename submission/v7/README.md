# v7 제출 모델 (코드·설정)

- `sub_v7.zip`과 **같은 코드와 설정**입니다. 모델 가중치(`models/siglip`, `models/blip`, `models/owl`)는 용량 때문에 git에 넣지 않았습니다.
- 전체 ZIP(가중치 포함, 1.0GB)은 이 저장소의 **Releases → `v7`** 에서 `sub_v7.zip`으로 받을 수 있습니다.
- 전략과 근거: [`docs/v7_strategy.md`](../../docs/v7_strategy.md)

## v4 대비 바뀐 것

| 항목 | v4 | v7 |
|---|---|---|
| BLIP이 다시 보는 후보 수 / 점수 비중 | 5장 / 0.2 | **10장 / 0.1** (v6부터) |
| BLIP 학습 | 형제 오답 7장, 문장 쪽만 | **검색 후보 오답 9장 + 이미지 인코더 마지막 2층도 학습** |
| 복수 관계 점검 ("두 X가 한 Y 위") | 없음 | **추가** (팀원 파서·판정 규칙, 검출기는 OWL) |
| 시간 안전장치 | 남은 작업량으로 OWL 시간 예측 | **경과 시간만** (모든 질의에 동일) |

## 파일

| 파일 | 내용 |
|---|---|
| `model.py` | `load_model` / `predict`: SigLIP 검색 → 조건 분해 재정렬 → BLIP ITM → OWL 위치 검증 → 복수 관계 점검 |
| `haiku/siblings.py` | 규칙 파서 (문장 → 물체·속성·관계) |
| `haiku/rerank.py` | 조건 12종 추출, 대조 문구, 조건 재정렬 |
| `haiku/owl.py` | OWL 관계 규칙 (좌우·상하), 여러 상자 검출 |
| `haiku/plural.py` | 복수 주체 관계 파서와 상자 판정 (팀원 원 실험 코드, 검출기 연결부만 OWL용) |
| `models/haiku_config.json` | 설정값 (K=10, 조건 비중 3.2, BLIP 10장·0.1, OWL 0.1/0.06, 복수 관계 확신도 0.2) |
| `requirements.txt` | 서버 설치 패키지 (== 버전 고정) |
