"""Qwen3-VL-Reranker 재정렬 측정 (그룹 분할 검증 800쌍).

방식은 팀원 group22의 l-resort-v1.ipynb를 옮긴 것이다:
1단계 임베딩 점수로 상위 K개를 뽑고, 그 후보만 VLM에게 "yes/no" 로짓 차이로 다시 점수화한다.
최종 점수 = z(1단계) + alpha * z(VLM), 1단계 1-2위 점수 차(margin)가 작은 질의만 재정렬.
각 질의는 자기 문장과 후보만 쓴다(다른 질의 결과를 쓰지 않음).

사용(Colab): exec(open('/content/vlm_rerank_eval.py').read()); run(ckpt='/content/g_p8', data=DATA)
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

KMAX = 4
RES = 320
BATCH = 16
JUDGE_SYS = ('Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
             'Note that the answer can only be "yes" or "no".')
INSTR = "<Instruct>: Retrieve images that match the user's description exactly.<Query>:"


def point(rank):
    return 0.0 if rank > 5 else 0.5 * ((rank == 1) + 1 / np.log2(rank + 1))


def stage1(ckpt, rows, dev="cuda"):
    from transformers import AutoProcessor, SiglipModel
    proc = AutoProcessor.from_pretrained(ckpt)
    net = SiglipModel.from_pretrained(ckpt).to(dev).half().eval()
    imgs = [Image.open(r["image_path"]).convert("RGB") for r in rows]
    with torch.inference_mode():
        iv = []
        for s in range(0, len(rows), 64):
            px = proc(images=imgs[s:s + 64], return_tensors="pt").pixel_values.to(dev).half()
            iv.append(torch.nn.functional.normalize(net.get_image_features(pixel_values=px).pooler_output.float(), dim=-1))
        tv = []
        for s in range(0, len(rows), 128):
            t = proc(text=[r["prompts"] for r in rows[s:s + 128]], padding="max_length", truncation=True,
                     return_tensors="pt").to(dev)
            tv.append(torch.nn.functional.normalize(net.get_text_features(**t).pooler_output.float(), dim=-1))
    S = (torch.cat(tv) @ torch.cat(iv).T).cpu()
    del net
    torch.cuda.empty_cache()
    return S, imgs


def run(ckpt, data, dev="cuda"):
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.57.0"], check=False)
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    data = Path(data)
    allrows = [json.loads(l) for l in open(data / "prompts.jsonl")]
    for r in allrows:
        r["image_path"] = str(data / "images" / ("%s.png" % r["id"]))
    byid = {r["id"]: r for r in allrows}
    ids = json.load(open(Path(ckpt) / "valid_ids.json"))
    rows = [byid[i] for i in ids]
    n = len(rows)
    S, imgs = stage1(ckpt, rows, dev)
    rank1 = ((S > S.diag()[:, None]).sum(1) + 1).numpy()
    topv, topi = S.topk(KMAX, dim=1)
    margin = (topv[:, 0] - topv[:, 1]).numpy()
    base = float(np.mean([point(int(r)) for r in rank1]))
    print("1단계 점수 %.4f | R@1 %.4f | 상위 2/3/4 안 %s" % (
        base, (rank1 == 1).mean(), " / ".join("%.3f" % (rank1 <= k).mean() for k in (2, 3, 4))), flush=True)

    name = "Qwen/Qwen3-VL-Reranker-2B"
    proc2 = AutoProcessor.from_pretrained(name)
    proc2.tokenizer.padding_side = "left"
    vocab = proc2.tokenizer.get_vocab()
    YES = [vocab[w] for w in ("yes", "Yes") if w in vocab]
    NO = [vocab[w] for w in ("no", "No") if w in vocab]
    m = Qwen3VLForConditionalGeneration.from_pretrained(name, dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
    W = m.lm_head.weight[YES + NO].float()
    small = [im.resize((RES, RES), Image.BICUBIC) for im in imgs]

    def conv(text, img):
        user = {"role": "user", "content": [{"type": "text", "text": INSTR}, {"type": "text", "text": text},
                                            {"type": "text", "text": "\n<Document>:"}, {"type": "image", "image": img}]}
        return [{"role": "system", "content": [{"type": "text", "text": JUDGE_SYS}]}, user]

    pairs = [(a, b) for a in range(n) for b in range(KMAX)]
    R = torch.zeros(len(pairs))
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, len(pairs), BATCH):
            ch = pairs[s:s + BATCH]
            inp = proc2.apply_chat_template([conv(rows[a]["prompts"], small[int(topi[a, b])]) for a, b in ch],
                                            tokenize=True, add_generation_prompt=True, return_dict=True,
                                            padding=True, return_tensors="pt").to(dev)
            lg = m.model(**inp).last_hidden_state[:, -1].float() @ W.T
            R[s:s + len(ch)] = (torch.logsumexp(lg[:, :len(YES)], 1) - torch.logsumexp(lg[:, len(YES):], 1)).cpu()
            if (s // BATCH) % 20 == 0:
                print("  VLM %d/%d  %.0f초" % (s, len(pairs), time.time() - t0), flush=True)
    torch.cuda.synchronize()
    pps = len(pairs) / (time.time() - t0)
    R = R.view(n, KMAX)
    print("VLM 속도 %.1f 쌍/초 (T4, %dpx, 배치 %d)" % (pps, RES, BATCH), flush=True)
    torch.save({"R": R, "topi": topi, "topv": topv, "rank1": rank1, "pps": pps}, "/content/vlm_scores.pt")

    def z(x):
        return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)

    def after(K, gate, alpha):
        f = z(topv[:, :K]) + alpha * z(R[:, :K])
        order = f.argsort(1, descending=True)
        new = rank1.copy()
        for a in np.where(gate & (rank1 <= K))[0]:
            new[a] = (topi[a, :K][order[a]] == a).nonzero()[0, 0].item() + 1
        return new

    print("\n설정 | 재정렬 비율 | 점수 | R@1 | 16k·8k 추가시간 추정(T4, 초)")
    for K in (2, 3, 4):
        for q in (1.0, 0.5, 0.25):
            gate = margin <= np.quantile(margin, q)
            for alpha in (0.5, 1, 2, 4):
                new = after(K, gate, alpha)
                sc = float(np.mean([point(int(r)) for r in new]))
                extra = 8000 * K * gate.mean() / pps
                print("K=%d 비율 %3.0f%% alpha %.1f | %.4f | %.4f | +%.0f초" % (
                    K, 100 * gate.mean(), alpha, sc, (new == 1).mean(), extra), flush=True)
