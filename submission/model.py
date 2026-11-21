"""평가 서버가 불러오는 검색 모델: 파인튜닝한 SigLIP + 파서 관계 문구 리랭킹.

점수 = z(전체 문장) + LAM * mean(z(관계 문구)). z는 쿼리(문구)마다 후보 풀 안에서만 구한다.
다른 쿼리의 예측은 쓰지 않는다.
"""
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.nn.functional import normalize
from transformers import AutoProcessor, SiglipModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_prompt import parse_prompt  # noqa: E402

LAM = 0.5

NUM = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
PRED = {"left_of": "to the left of", "right_of": "to the right of", "above": "above",
        "below": "below", "next_to": "next to", "on": "resting on", "inside": "inside",
        "in_front_of": "in front of", "behind": "behind", "holds": "holding",
        "wears": "wearing", "rides": "riding"}


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
    """프롬프트 -> 관계 문구 목록. 파싱에 실패하면 빈 목록(전체 문장 점수만 사용)."""
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
    checkpoint_dir = Path(model_dir) / "models"
    network = SiglipModel.from_pretrained(checkpoint_dir, local_files_only=True).to(device)
    if network.device.type == "cuda":
        network.half()
    processor = AutoProcessor.from_pretrained(checkpoint_dir, local_files_only=True)
    return network.eval(), processor


def read_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def tokenize(processor, prompts):
    return processor(text=prompts, padding="max_length", truncation=True, return_tensors="pt")


def _zrow(M):
    return (M - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-6)


@torch.inference_mode()
def predict(model, queries, candidates, top_k=5):
    network, processor = model
    network.eval()
    dev = network.device

    def encode_text(texts, batch_size=256):
        outs = []
        for s in range(0, len(texts), batch_size):
            tokens = tokenize(processor, texts[s:s + batch_size]).to(dev)
            features = network.get_text_features(**tokens).pooler_output
            outs.append(normalize(features.float(), dim=-1))
        return torch.cat(outs)

    # 이미지 임베딩: 후보 이미지마다 한 번만 계산
    vecs = []
    for s in range(0, len(candidates), 32):
        images = [read_image(item["image_path"]) for item in candidates[s:s + 32]]
        pixels = processor(images=images, return_tensors="pt").pixel_values
        pixels = pixels.to(device=dev, dtype=network.dtype)
        features = network.get_image_features(pixel_values=pixels).pooler_output
        vecs.append(normalize(features.float(), dim=-1))
    image_vectors = torch.cat(vecs)
    image_ids = [item["image_id"] for item in candidates]

    # 쿼리 임베딩 + 관계 문구
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

    predictions = {}
    for s in range(0, len(queries), 256):
        batch = queries[s:s + 256]
        scores = _zrow(text_vectors[s:s + 256] @ image_vectors.T)
        if part_z is not None:
            weights = torch.zeros(len(batch), n_unique, dtype=torch.half)
            for r, ps in enumerate(parts[s:s + 256]):
                if ps:
                    idx = torch.tensor([pid[p] for p in ps])
                    weights[r].index_add_(0, idx, torch.full((len(ps),), 1.0 / len(ps), dtype=torch.half))
            scores = scores + LAM * (weights.to(dev) @ part_z).float()
        top = scores.topk(min(top_k, scores.shape[1]), dim=1).indices.tolist()
        for query, row in zip(batch, top):
            predictions[query["prompt_id"]] = [image_ids[j] for j in row]
    return predictions
