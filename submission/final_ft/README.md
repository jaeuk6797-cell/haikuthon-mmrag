# 최종 후보: SigLIP 미세조정(FT) 검색 → 조건B → 색(물체별 crop) → OWL 관계 (BLIP 선택)

가중치는 저장소에 없다(.gitignore). 제출 폴더 조립은 `experiments_colab/asm.py` 참고.

- `models/siglip`: FT 가중치(`experiments_colab/exp_ftall.py`, V_ALL=1이면 4000쌍 전부 학습, fp16 safetensors)로 `google/siglip-base-patch16-384` 파일 교체
- `models/siglip_zs`: 제로샷 `google/siglip-base-patch16-384` (색 crop 검사는 제로샷으로 계산. 실험이 그렇게 검증됨)
- `models/owl`: `google/owlvit-base-patch32`
- (선택) `models/blip` + `models/blip/s0_weights.safetensors`: 공개 BLIP-ITM 위에 팀원 S0 가중치를 덮어씀. `models/haiku_config.json`의 `itm.use`를 true로
- 검증(그룹 분할 seed 42, 검증 800 풀): FT 단독 0.5953, FT+조건B(lam1)+색(mu0.5)+OWL(top3) 0.7119, 학습형 상위5 0.7203. E(제로샷 기반) 0.6175. 결과 원문은 `analysis/results/`
