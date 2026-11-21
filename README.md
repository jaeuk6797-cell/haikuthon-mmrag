# HAIKUTHON 2026 - Multimodal RAG

AIKU x HAI 연합 데이터톤. 영어 텍스트 설명(query)이 주어지면 후보 이미지 중 정답 1장을 찾아 상위 5장을 순서대로 제출한다.

## 과제 요약
- Text -> Image 검색. 각 query의 정답 이미지는 정확히 1장
- Train: 텍스트-이미지 4,000쌍 (HuggingFace `KimWooJin/26-haikuthon-train`)
- Test (비공개): 텍스트 8,000개, 이미지 16,000장
- 쿼리는 개수, 색, 공간 관계를 포함한 구성적 묘사 (예: "One green bonsai tree resting on one green shelf is next to one blue chair.")

## 채점
- 쿼리별 점수 = 0.5 x (Hit-Ratio@1 + NDCG@5), 전체 평균
- 정답 순위별 점수: 1위 1.000 / 2위 0.315 / 3위 0.250 / 4위 0.215 / 5위 0.193 / 상위 5개 밖 0
- 최종 점수 = 리더보드 70% (Public 40% + Private 60%) + 발표 심사 30%

## 제출 제약
- 모델(model.py + safetensors) zip 5GB 이하
- 추론 시간 최대 15분 (모델 로딩 제외), NVIDIA L4 24GB x 10
- 팀당 최대 25회 제출
- 금지: 외부 API 호출, 다른 쿼리의 예측으로 현재 쿼리 예측을 보정하는 행위

## 일정
- 2026-11-22 08:00 모델 제출 마감 및 리더보드 마감
- 결선 10팀 발표 (팀당 8분)

## 폴더 구조
- `analysis/`: 데이터 분석 스크립트와 결과
- `src/`: 모델과 학습 코드
- `submission/`: 제출용 model.py와 패키징 스크립트
- `experiments.md`: 실험 기록 (ablation, 실패한 시도 포함)

데이터 파일과 모델 가중치는 저장소에 올리지 않는다 (`.gitignore` 참고).
