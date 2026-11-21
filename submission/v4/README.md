# v4 제출 모델 (코드·설정)

- 제출한 `sub_v4.zip`과 **같은 코드와 설정**입니다. 모델 가중치(`models/siglip`, `models/blip`, `models/owl`)는 용량 때문에 git에 넣지 않았습니다.
- 전체 ZIP(가중치 포함, 1.0GB)은 이 저장소의 **Releases → `v4`** 에서 `sub_v4.zip`으로 받을 수 있습니다.
- 전략과 근거: [`docs/v4_strategy.md`](../../docs/v4_strategy.md)

| 파일 | 내용 |
|---|---|
| `model.py` | `load_model` / `predict`: SigLIP 검색 → 조건 분해 재정렬 → BLIP ITM → OWL-ViT 위치 검증 |
| `haiku/siblings.py` | 규칙 파서 (문장 → 물체·속성·관계) |
| `haiku/rerank.py` | 조건 12종 추출, 대조 문구, 조건 재정렬 |
| `haiku/owl.py` | OWL 관계 규칙 (좌우·상하) |
| `models/haiku_config.json` | 설정값 (λ=3.2, BLIP k=5 λ=0.2, OWL 기준 0.1/0.06, 시간 예산 830초) |
| `requirements.txt` | 서버 설치 패키지 (== 버전 고정) |
