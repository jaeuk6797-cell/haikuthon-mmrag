"""평가 서버가 불러오는 검색 모델: 파인튜닝한 SigLIP + 관계 문구 리랭킹 + Qwen3-VL-Reranker 재정렬.

1) 점수 = z(전체 문장) + LAM * mean(z(관계 문구)). z는 쿼리마다 후보 풀 안에서만 구한다.
2) 1위와 2위의 원본 코사인 차이(margin)가 임계값보다 작은 질의만, 그 두 후보를 VLM에게 물어
   VLM이 2위를 더 낫다고 하면 순서를 바꾼다.
각 질의는 자기 문장과 자기 후보만 쓴다. 다른 질의의 예측을 쓰지 않는다.
"""
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from torch.nn.functional import normalize
from transformers import AutoProcessor, SiglipModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_prompt import parse_prompt  # noqa: E402

LAM = 0.5
VLM_FRAC = 0.2       # 1위-2위 원본 코사인 차이가 작은 쪽부터 이 비율의 질의만 VLM 재정렬 (풀 크기와 무관하게 개수가 정해짐)
TIME_BUDGET = 780.0  # predict 시작 후 이 초를 넘기면 남은 VLM 재정렬은 건너뜀 (제한 900초 보호)
VLM_CHUNK = 128      # VLM 재정렬을 이 질의 수 단위로 처리하며 시간 확인
VLM_RES = 320
VLM_BATCH = 16
LAST = {}  # 디버그/검증용 (질의별 margin, VLM 점수)

NUM = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
PRED = {"left_of": "to the left of", "right_of": "to the right of", "above": "above",
        "below": "below", "next_to": "next to", "on": "resting on", "inside": "inside",
        "in_front_of": "in front of", "behind": "behind", "holds": "holding",
        "wears": "wearing", "rides": "riding"}

JUDGE_SYS = ('Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
             'Note that the answer can only be "yes" or "no".')
INSTR = "<Instruct>: Retrieve images that match the user's description exactly.<Query>:"


def _plural(w):
    if re.search(r"(s|x|ch|sh)$", w):
        return w + "es"
    if re.search(r"[^aeiou]y$", w):
        return w[:-1] + "ies"
    return w + "s"


def _obj_phrase(o):
    attrs = [v for vs in o["attrs"].values() for v in vs]
    n = o["count"]
    if n and n > 1:
        return " ".join([NUM.get(n, str(n))] + attrs + [_plural(o["label"])])
    return " ".join(["a"] + attrs + [o["label"]])


def _relation_phrases(text):
    try:
        p = parse_prompt(text)
        objs = p["objects"]
        out = []
        for r in p["relations"]:
            pred = PRED.get(r["pred"])
            if pred is None:
                continue
            out.append("%s %s %s" % (_obj_phrase(objs[r["subj"]]), pred, _obj_phrase(objs[r["obj"]])))
        return out
    except Exception:
        return []


def load_model(model_dir, device):
    root = Path(model_dir)
    network = SiglipModel.from_pretrained(root / "models", local_files_only=True).to(device)
    if network.device.type == "cuda":
        network.half()
    processor = AutoProcessor.from_pretrained(root / "models", local_files_only=True)
    vlm = vproc = None
    vdir = root / "vlm"
    if vdir.exists():
        from transformers import Qwen3VLForConditionalGeneration
        vproc = AutoProcessor.from_pretrained(vdir, local_files_only=True)
        vproc.tokenizer.padding_side = "left"
        vlm = _load_vlm_int8(vdir, device)
    return network.eval(), processor, vlm, vproc


def _load_vlm_int8(vdir, device):
    """vdir/model_int8.safetensors (2D 큰 가중치는 행별 int8 + 스케일, 나머지는 fp16)를 fp16으로 복원해 로드.
    파일에서 한 텐서씩 읽어 바로 모델 파라미터에 복사하므로 GPU 메모리는 모델 크기 + 텐서 하나분만 쓴다."""
    from safetensors import safe_open
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    cfg = AutoConfig.from_pretrained(vdir, local_files_only=True)
    cfg._attn_implementation = "sdpa"
    with torch.device(device):
        vlm = Qwen3VLForConditionalGeneration._from_config(cfg, dtype=torch.float16)
    state = vlm.state_dict()
    seen, unexpected = set(), []
    with safe_open(str(Path(vdir) / "model_int8.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith(".s"):
                continue
            if k.endswith(".q"):
                name = k[:-2]
                t = (f.get_tensor(k).to(device).float() * f.get_tensor(name + ".s").to(device)).half()
            else:
                name = k
                t = f.get_tensor(k).to(device)
            if name in state:
                state[name].copy_(t)
                seen.add(name)
            else:
                unexpected.append(name)
            del t
    vlm.tie_weights()
    missing = [n for n in state if n not in seen and "lm_head" not in n]
    if missing or unexpected:
        raise RuntimeError("VLM 가중치 불일치: missing=%s unexpected=%s" % (missing[:5], unexpected[:5]))
    del state
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return vlm.eval()


def read_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def tokenize(processor, prompts):
    return processor(text=prompts, padding="max_length", truncation=True, return_tensors="pt")


def _zrow(M):
    return (M - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-6)


def _vlm_scores(vlm, vproc, pairs, paths, dev):
    """pairs: [(질의 문장, image_path)] -> yes/no 로짓 차이."""
    vocab = vproc.tokenizer.get_vocab()
    yes = [vocab[w] for w in ("yes", "Yes") if w in vocab]
    no = [vocab[w] for w in ("no", "No") if w in vocab]
    W = vlm.lm_head.weight[yes + no].float()

    def load(p):
        with Image.open(p) as im:
            return im.convert("RGB").resize((VLM_RES, VLM_RES), Image.BICUBIC)

    uniq = sorted(set(paths))
    with ThreadPoolExecutor(4) as ex:
        imgs = dict(zip(uniq, ex.map(load, uniq)))
    out = torch.zeros(len(pairs))
    for s in range(0, len(pairs), VLM_BATCH):
        ch = pairs[s:s + VLM_BATCH]
        convs = []
        for text, p in ch:
            user = {"role": "user", "content": [{"type": "text", "text": INSTR}, {"type": "text", "text": text},
                                                {"type": "text", "text": "\n<Document>:"},
                                                {"type": "image", "image": imgs[p]}]}
            convs.append([{"role": "system", "content": [{"type": "text", "text": JUDGE_SYS}]}, user])
        inp = vproc.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                        padding=True, return_tensors="pt").to(dev)
        lg = vlm.model(**inp).last_hidden_state[:, -1].float() @ W.T
        out[s:s + len(ch)] = (torch.logsumexp(lg[:, :len(yes)], 1) - torch.logsumexp(lg[:, len(yes):], 1)).cpu()
    return out


@torch.inference_mode()
def predict(model, queries, candidates, top_k=5):
    t_start = time.time()
    network, processor, vlm, vproc = model
    network.eval()
    dev = network.device

    def encode_text(texts, batch_size=256):
        outs = []
        for s in range(0, len(texts), batch_size):
            tokens = tokenize(processor, texts[s:s + batch_size]).to(dev)
            features = network.get_text_features(**tokens).pooler_output
            outs.append(normalize(features.float(), dim=-1))
        return torch.cat(outs)

    vecs = []
    for s in range(0, len(candidates), 32):
        images = [read_image(item["image_path"]) for item in candidates[s:s + 32]]
        pixels = processor(images=images, return_tensors="pt").pixel_values
        pixels = pixels.to(device=dev, dtype=network.dtype)
        features = network.get_image_features(pixel_values=pixels).pooler_output
        vecs.append(normalize(features.float(), dim=-1))
    image_vectors = torch.cat(vecs)
    image_ids = [item["image_id"] for item in candidates]
    image_paths = [item["image_path"] for item in candidates]

    text_vectors = encode_text([q["prompt"] for q in queries])
    parts = [_relation_phrases(q["prompt"]) for q in queries]
    unique = sorted({p for ps in parts for p in ps})
    pid = {p: i for i, p in enumerate(unique)}
    n_unique = len(unique)

    part_z = None
    if n_unique:
        part_vectors = encode_text(unique)
        part_z = torch.empty(n_unique, image_vectors.shape[0], dtype=torch.half, device=dev)
        for s in range(0, n_unique, 1024):
            part_z[s:s + 1024] = _zrow(part_vectors[s:s + 1024] @ image_vectors.T).half()

    tops, margins = [], []
    for s in range(0, len(queries), 256):
        batch = queries[s:s + 256]
        raw = text_vectors[s:s + 256] @ image_vectors.T
        scores = _zrow(raw)
        if part_z is not None:
            weights = torch.zeros(len(batch), n_unique, dtype=torch.half)
            for r, ps in enumerate(parts[s:s + 256]):
                if ps:
                    idx = torch.tensor([pid[p] for p in ps])
                    weights[r].index_add_(0, idx, torch.full((len(ps),), 1.0 / len(ps), dtype=torch.half))
            scores = scores + LAM * (weights.to(dev) @ part_z).float()
        top = scores.topk(min(top_k, scores.shape[1]), dim=1).indices
        r1 = raw.gather(1, top[:, :1])[:, 0]
        r2 = raw.gather(1, top[:, 1:2])[:, 0]
        margins.extend((r1 - r2).tolist())
        tops.extend(top.tolist())

    # VLM 재정렬: 1-2위 원본 코사인 차이가 작은 질의부터, 개수 상한과 시간 상한 안에서만
    swapped = 0
    LAST.clear()
    LAST["margin"] = margins
    if vlm is not None and VLM_FRAC > 0:
        order = sorted(range(len(queries)), key=lambda i: margins[i])
        gate = order[:max(1, int(round(VLM_FRAC * len(queries))))]
        vscores, done = {}, 0
        chunk_time = 0.0
        for s in range(0, len(gate), VLM_CHUNK):
            elapsed = time.time() - t_start
            if elapsed + chunk_time > TIME_BUDGET:
                break
            t_c = time.time()
            ids = gate[s:s + VLM_CHUNK]
            pairs, paths = [], []
            for i in ids:
                for j in tops[i][:2]:
                    pairs.append((queries[i]["prompt"], image_paths[j]))
                    paths.append(image_paths[j])
            sc = _vlm_scores(vlm, vproc, pairs, paths, dev).view(-1, 2)
            for k, i in enumerate(ids):
                vscores[i] = sc[k].tolist()
                if sc[k, 1] > sc[k, 0]:
                    tops[i][0], tops[i][1] = tops[i][1], tops[i][0]
                    swapped += 1
            done += len(ids)
            chunk_time = max(chunk_time, time.time() - t_c)
        LAST["vlm"] = vscores
        LAST["gated"], LAST["done"], LAST["swapped"] = len(gate), done, swapped
        LAST["elapsed"] = time.time() - t_start

    return {q["prompt_id"]: [image_ids[j] for j in row] for q, row in zip(queries, tops)}
