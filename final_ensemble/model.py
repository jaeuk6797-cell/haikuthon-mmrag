"""Offline HAIKUTHON retriever and sibling-trained reranker.

Call prepare_gallery once when querying or rescoring multiple candidate lists.
All model paths are under model_dir. No training or outside service is used.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn.functional import normalize


ONE_READ_WORKERS = 4




def _features(output):
    values = output if torch.is_tensor(output) else output.pooler_output
    return normalize(values.float(), dim=-1)




def _blip_text_tokens(processor, texts, device):
    # BLIP ITM uses the same right-padded BERT sequence for every image of a query.
    if processor.tokenizer.padding_side != "right":
        raise ValueError("BLIP ITM expects right padding.")
    ids, masks = [], []
    for start in range(0, len(texts), 128):
        tokens = processor(text=texts[start:start + 128], padding="max_length", truncation=True,
                           max_length=128, return_tensors="pt")
        ids.append(tokens["input_ids"])
        masks.append(tokens["attention_mask"])
    ids = torch.cat(ids)
    masks = torch.cat(masks)
    lengths = masks.sum(1).numpy().astype(np.int64)
    return ids.to(device), masks.to(device), lengths




def _one_read_spec(processor, name, device, dtype):
    # Use the exact archived HF rescale/normalize values on all 256 possible RGB
    # byte values. This avoids substituting a different GPU arithmetic sequence.
    ip = processor.image_processor
    expected = {"siglip": ("SiglipImageProcessor", 224), "blip": ("BlipImageProcessor", 384)}
    expected_class, expected_size = expected[name]
    if type(ip).__name__ != expected_class or not ip.do_resize:
        raise ValueError(f"Unexpected {name} image processor: {type(ip).__name__}")
    size = (int(ip.size["width"]), int(ip.size["height"]))
    if size != (expected_size, expected_size):
        raise ValueError(f"Unexpected {name} image size: {size}")
    ramp = np.repeat(np.arange(256, dtype=np.uint8).reshape(16, 16, 1), 3, axis=2)
    with Image.fromarray(ramp) as im:
        values = ip(images=im, do_resize=False, return_tensors="pt")["pixel_values"]
    if values.shape != (1, 3, 16, 16):
        raise ValueError(f"Unexpected {name} preprocessing shape: {tuple(values.shape)}")
    return (size, ip.resample), values[0].reshape(3, 256).contiguous().to(device=device, dtype=dtype)


def _one_read_fill(path, specs, targets):
    # Both independent Pillow resizes start from the same decoded original RGB.
    with Image.open(path) as opened, opened.convert("RGB") as rgb:
        for (size, resample), target in zip(specs, targets):
            with rgb.resize(size, resample=resample) as resized:
                np.copyto(target, np.asarray(resized).transpose(2, 0, 1))


def _one_read_batches(paths, specs, workers, pin):
    # At most two batches have outstanding CPU work. No CUDA process forks and
    # no whole-gallery copy in host RAM; every worker writes to its own row.
    total = len(paths)
    with (ThreadPoolExecutor(max_workers=workers) if workers else nullcontext(None)) as pool:
        pending = deque()
        def prepare(start):
            count = min(32, total - start)
            buffers = [torch.empty((count, 3, size[1], size[0]), dtype=torch.uint8, pin_memory=pin)
                       for size, _ in specs]
            host = [buf.numpy() for buf in buffers]
            futures = []
            for row in range(count):
                targets = [array[row] for array in host]
                if pool is None:
                    _one_read_fill(paths[start + row], specs, targets)
                else:
                    futures.append(pool.submit(_one_read_fill, paths[start + row], specs, targets))
            return start, buffers, futures

        starts = iter(range(0, total, 32))
        for _ in range(2 if workers else 1):
            start = next(starts, None)
            if start is not None:
                pending.append(prepare(start))
        while pending:
            start, (sig, blip), futures = pending.popleft()
            for future in futures:
                future.result()
            yield start, sig, blip
            following = next(starts, None)
            if following is not None:
                pending.append(prepare(following))


def _one_read_pixels(raw, lut):
    return torch.stack([lut[channel][raw[:, channel].long()] for channel in range(3)], dim=1)


def _one_read_gallery(bundle, paths):
    device, dtype = bundle["device"], bundle["dtype"]
    sig_spec, sig_lut = _one_read_spec(bundle["siglip_processor"], "siglip", device, dtype)
    blip_spec, blip_lut = _one_read_spec(bundle["blip_processor"], "blip", device, dtype)
    width, height = blip_spec[0]
    raw = torch.empty((len(paths), 3, height, width), dtype=torch.uint8, device=device)
    pin = device.type == "cuda"
    vectors = []
    for start, sig, blip in _one_read_batches(paths, (sig_spec, blip_spec), ONE_READ_WORKERS, pin):
        raw[start:start + len(blip)].copy_(blip, non_blocking=pin)
        pixels = _one_read_pixels(sig.to(device=device, non_blocking=pin), sig_lut)
        vectors.append(_features(bundle["siglip"].get_image_features(pixel_values=pixels)))
    return torch.cat(vectors), raw, blip_lut


class PairHead(nn.Module):
    """Compare a query with interacting image tokens in both input orders."""

    def __init__(self, width=96, heads=4, input_width=768):
        super().__init__()
        if width < 1 or heads < 1 or width % heads:
            raise ValueError("pair_dim must be positive and divisible by pair_heads")
        self.image_in = nn.Sequential(nn.LayerNorm(input_width), nn.Linear(input_width, width))
        self.text_in = nn.Sequential(nn.LayerNorm(input_width), nn.Linear(input_width, width))
        axis = torch.linspace(-1, 1, 6)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        xy = torch.stack([xx.flatten(), yy.flatten(), torch.zeros(36)], dim=1)
        self.register_buffer("locations", torch.cat([torch.tensor([[0., 0., 1.]]), xy]))
        self.location_in = nn.Linear(3, width, bias=False)
        self.cross_image = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.evidence = nn.Sequential(
            nn.Linear(width * 4, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.query_reads_pair = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.text_gate = nn.Linear(width, 1)
        self.token_score = nn.Sequential(
            nn.Linear(width * 3, width), nn.GELU(), nn.Linear(width, 1)
        )

    def forward(self, text, text_mask, left, right):
        if left.ndim != 3 or left.shape != right.shape or left.size(1) != 37:
            raise ValueError("Each image needs a CLS token followed by a row-major 6x6 grid.")
        text = self.text_in(text.float())
        text_mask = text_mask.bool()
        n = left.size(0)
        # Evaluate both ordered pairs with identical functions and parameters.
        own = self.image_in(torch.cat((left, right)).float()) + self.location_in(self.locations)
        other = torch.cat((own[n:], own[:n]))
        matched, _ = self.cross_image(own, other, other, need_weights=False)
        # Two kinds of differences: corresponding locations and learned cross-image matches.
        evidence = self.evidence(torch.cat((own, own - other, own - matched, own * matched), dim=-1))
        query = torch.cat((text, text))
        read, _ = self.query_reads_pair(query, evidence, evidence, need_weights=False)
        token_scores = self.token_score(torch.cat((query, read, query * read), dim=-1)).squeeze(-1)
        gate = self.text_gate(text).squeeze(-1).masked_fill(~text_mask, -torch.inf).softmax(dim=1)
        # Only valid tokens participate. Feature preparation guarantees a valid CLS per query.
        scores = (token_scores * torch.cat((gate, gate))).sum(dim=1)
        return scores[:n] - scores[n:]


def _overlay(text_encoder, itm_head, overlay_file, arm):
    from safetensors import safe_open

    params = {"text_encoder." + name: tensor for name, tensor in text_encoder.named_parameters()}
    params.update({"itm_head." + name: tensor for name, tensor in itm_head.named_parameters()})
    if arm == "C":
        layers = text_encoder.encoder.layer
        prefix = f"text_encoder.encoder.layer.{len(layers)-1}."
        blocks = dict(layers[-1].named_children())
        cross = next(name for name in ("crossattention", "cross_attention", "cross_attn") if name in blocks)
        params = {k: v for k, v in params.items() if k.startswith((prefix + cross + ".", "itm_head."))}
    with torch.no_grad(), safe_open(str(overlay_file), framework="pt", device="cpu") as weights:
        if set(weights.keys()) != set(params):
            raise ValueError(f"Unexpected or incomplete {arm} BLIP weights.")
        for name, target in params.items():
            value = weights.get_tensor(name)
            if value.shape != target.shape or not value.is_floating_point():
                raise ValueError(f"Invalid {arm} parameter {name}.")
            target.copy_(value)


def load_model(model_dir, device, *, arms=None, use_pair=None):
    """Load only selected BLIP scorers. Default is S0 unless configured in the package."""
    from safetensors.torch import load_file
    from transformers import AutoProcessor, BlipForImageTextRetrieval, BlipProcessor, SiglipModel

    root = Path(model_dir)
    cfg_path = root / "ensemble_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    from_config = arms is None
    if arms is None:
        arms = cfg.get("arms", ["S0"])
    if isinstance(arms, str):
        arms = [arms]
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or set(arms) - {"A", "C", "S0", "S1_noaux"}:
        raise ValueError("Select distinct arms from A, S0, S1_noaux and C.")
    pair_enabled = cfg.get("use_pair", False) if use_pair is None else use_pair
    if type(pair_enabled) is not bool:
        raise ValueError("use_pair must be a bool.")
    device = torch.device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    models = root / "models"

    def load_net(cls, name):
        net = cls.from_pretrained(models / name, local_files_only=True, use_safetensors=True,
                                  torch_dtype=dtype)
        return net.requires_grad_(False).eval().to(device)

    siglip = load_net(SiglipModel, "siglip")
    blip = load_net(BlipForImageTextRetrieval, "blip")
    heads = {}
    for arm in arms:
        if arm == "A":
            heads[arm] = (blip.text_encoder, blip.itm_head)
            continue
        # In the single-arm default, directly overlay the public text tower in place.
        # Keep the public tower intact when comparing multiple models or running the pair head.
        clone = len(arms) > 1 or pair_enabled
        encoder = deepcopy(blip.text_encoder) if clone else blip.text_encoder
        itm = deepcopy(blip.itm_head) if clone else blip.itm_head
        _overlay(encoder, itm, models / "arms" / (arm + ".safetensors"), arm)
        heads[arm] = (encoder, itm)
    pair = None
    if pair_enabled:
        pair = PairHead()
        pair.load_state_dict(load_file(str(models / "pair_head.safetensors"), device="cpu"), strict=True)
        pair.requires_grad_(False).eval().to(device)
    return {
        "device": device, "dtype": dtype, "siglip": siglip, "blip": blip,
        "siglip_processor": AutoProcessor.from_pretrained(models / "siglip", local_files_only=True,
                                                          use_fast=False),
        "blip_processor": BlipProcessor.from_pretrained(models / "blip", local_files_only=True,
                                                       use_fast=False),
        "heads": heads, "pair": pair, "width": cfg.get("width", 8),
        "weights": cfg.get("weights") if from_config else None,
        "use_pair": pair_enabled, "_owner": object(),
    }


class _Gallery:
    def __init__(self, owner, ids, vectors, pixels, lut):
        self.owner = owner
        self.ids = ids
        self.index = {image_id: pos for pos, image_id in enumerate(ids)}
        self.vectors = vectors
        self.pixels = pixels
        self.lut = lut


def _width(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("Candidate count must be a nonnegative integer.")
    return int(value)


@torch.inference_mode()
def prepare_gallery(model, candidates):
    """Decode each input PNG once. Retain SigLIP vectors and resized byte images in memory."""
    if isinstance(candidates, _Gallery):
        if candidates.owner is not model["_owner"]:
            raise ValueError("Prepared gallery belongs to another loaded model instance.")
        return candidates
    ids = [item["image_id"] for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image_id.")
    if not ids:
        return _Gallery(model["_owner"], [], None, None, None)
    paths = [item["image_path"] for item in candidates]
    vectors, pixels, lut = _one_read_gallery(model, paths)
    return _Gallery(model["_owner"], ids, vectors, pixels, lut)


def _query_fields(queries):
    ids = [item["prompt_id"] for item in queries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate prompt_id.")
    return ids, [item["prompt"] for item in queries]


def _get_pools(model, texts, gallery, width, prescribed):
    pools, cosines = [], []
    for start in range(0, len(texts), 128):
        block = texts[start:start + 128]
        inputs = model["siglip_processor"](text=block, padding="max_length", truncation=True,
                                            return_tensors="pt")
        tokens = {k: inputs[k].to(model["device"]) for k in ("input_ids", "attention_mask") if k in inputs}
        features = _features(model["siglip"].get_text_features(**tokens))
        scores = features @ gallery.vectors.T
        if not torch.isfinite(scores).all().item():
            raise FloatingPointError("Non-finite SigLIP retrieval scores.")
        if prescribed is None:
            indices = torch.argsort(scores, descending=True, dim=1, stable=True)[:, :width]
            pools.extend(indices.cpu().tolist())
            cosines.extend(scores.gather(1, indices).cpu().tolist())
        else:
            for row, pool in enumerate(prescribed[start:start + len(block)]):
                pools.append(pool)
                cosines.append(scores[row, pool].cpu().tolist() if pool else [])
    return pools, cosines


@torch.inference_mode()
def retrieve_candidates(model, queries, gallery, top_k=8):
    """Return {prompt_id: [image_id, ...]} ordered by retriever cosine."""
    width = _width(top_k)
    qids, texts = _query_fields(queries)
    if not qids or not width:
        return {qid: [] for qid in qids}
    gallery = prepare_gallery(model, gallery)
    if not gallery.ids:
        return {qid: [] for qid in qids}
    rows, _ = _get_pools(model, texts, gallery, width, None)
    return {q: [gallery.ids[i] for i in row] for q, row in zip(qids, rows)}


def _pair_margins(model, gallery, pools, tokens, use_pair):
    ids, mask, lengths = tokens
    sizes = np.fromiter((len(row) for row in pools), dtype=np.int64)
    query_index = np.repeat(np.arange(len(pools)), sizes)
    flat = np.fromiter((i for row in pools for i in row), dtype=np.int64, count=int(sizes.sum()))
    unique, inverse = np.unique(flat, return_inverse=True)
    grouped = np.argsort(inverse, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(inverse))]
    device = model["device"]
    result = {arm: np.empty(len(flat), dtype=np.float32) for arm in model["heads"]}
    compact = (torch.empty((len(unique), 37, 768), device=device, dtype=torch.float16)
               if use_pair else None)
    for start in range(0, len(unique), 24):
        end = min(start + 24, len(unique))
        indices = torch.as_tensor(unique[start:end], device=device)
        pixels = _one_read_pixels(gallery.pixels.index_select(0, indices), gallery.lut)
        patches = model["blip"].vision_model(pixel_values=pixels, return_dict=True).last_hidden_state
        del pixels
        if use_pair:
            if patches.shape[1:] != (577, 768):
                raise ValueError("Pair head expects BLIP CLS and a 24x24 patch grid.")
            h = patches.to(torch.float16)
            compact[start:end, :1] = h[:, :1]
            cells = h[:, 1:].float().reshape(len(h), 6, 4, 6, 4, 768).mean(dim=(2, 4))
            compact[start:end, 1:] = cells.reshape(len(h), 36, 768)
        pairs = grouped[offsets[start]:offsets[end]]
        pairs = pairs[np.argsort(lengths[query_index[pairs]], kind="stable")]
        for offset in range(0, len(pairs), 32):
            group = pairs[offset:offset+32]
            rows = query_index[group]
            length = int(lengths[rows].max())
            q = torch.as_tensor(rows, device=device)
            img = torch.as_tensor(inverse[group] - start, device=device)
            vision = patches.index_select(0, img)
            vision_mask = torch.ones(vision.shape[:2], dtype=torch.long, device=device)
            text_ids = ids.index_select(0, q)[:, :length]
            text_mask = mask.index_select(0, q)[:, :length]
            for arm, (encoder, itm) in model["heads"].items():
                fused = encoder(input_ids=text_ids, attention_mask=text_mask,
                                encoder_hidden_states=vision, encoder_attention_mask=vision_mask,
                                return_dict=True)
                logits = itm(fused.last_hidden_state[:, 0]).float()
                result[arm][group] = (logits[:, 1] - logits[:, 0]).cpu().numpy()
    if not all(np.isfinite(values).all() for values in result.values()):
        raise FloatingPointError("Non-finite BLIP ITM margins.")
    return result, unique, compact


def _apply_pair(model, orders, pools, tokens, unique, compact):
    ids, masks, lengths = tokens
    eligible = [r for r, order in enumerate(orders) if len(order) >= 2]
    evidence = {}
    for offset in range(0, len(eligible), 32):
        rows = eligible[offset:offset+32]
        length = int(lengths[rows].max())
        text_mask = masks[rows, :length]
        public_text = model["blip"].text_encoder(input_ids=ids[rows, :length],
                                                attention_mask=text_mask, return_dict=True).last_hidden_state
        choices = [[pools[r][orders[r][j]] for j in (0, 1)] for r in rows]
        pair_ids = torch.as_tensor(np.searchsorted(unique, choices), device=compact.device)
        # Match the frozen feature bank used in training, including its storage precision.
        logit = model["pair"](public_text.half().float(), text_mask.bool(),
                              compact[pair_ids[:, 0]].float(), compact[pair_ids[:, 1]].float())
        if not torch.isfinite(logit).all().item():
            raise FloatingPointError("Non-finite Top-2 comparator logits.")
        for r, probability in zip(rows, torch.sigmoid(-logit).cpu().tolist()):
            first_col, second_col = orders[r][:2]
            evidence[r] = {first_col: (1, 1.0-probability), second_col: (2, probability)}
            if probability >= 0.75:
                orders[r][[0, 1]] = orders[r][[1, 0]]
    return evidence


def _mixture(model, weights):
    if weights is None:
        weights = model["weights"]
    if weights is None:
        weights = {arm: 1.0 for arm in model["heads"]}
    if not isinstance(weights, dict) or not weights or set(weights) - set(model["heads"]):
        raise ValueError("weights must map loaded arm names to nonnegative finite weights.")
    values = {arm: float(w) for arm, w in weights.items()}
    if any(not math.isfinite(w) or w < 0 for w in values.values()):
        raise ValueError("Invalid arm weight.")
    total = sum(values.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("A finite positive total weight is required.")
    return {arm: w / total for arm, w in values.items() if w}


@torch.inference_mode()
def score_candidates(model, queries, candidates_or_gallery, candidate_lists=None, *,
                     width=None, weights=None, use_pair=None):
    """Return ranked evidence for a common per-query pool.

    candidate_lists may map each prompt ID to distinct image IDs, preserving its supplied
    tie order. Without it, select the retriever's top width, normally eight. A single-arm
    score is its raw BLIP margin. Multiple arms combine per-query population z-scores.
    Raw margins and cosine remain available after the optional pair head changes ranks.
    """
    qids, texts = _query_fields(queries)
    width = _width(model["width"] if width is None else width)
    mix = _mixture(model, weights)
    pair_enabled = model["use_pair"] if use_pair is None else use_pair
    if type(pair_enabled) is not bool or (pair_enabled and model["pair"] is None):
        raise ValueError("Load the comparator with use_pair=True before using it.")
    if candidate_lists is not None and set(candidate_lists) != set(qids):
        raise ValueError("candidate_lists must have exactly the supplied prompt IDs.")
    empty = {qid: [] for qid in qids}
    if not qids or (candidate_lists is None and width == 0):
        return empty
    gallery = prepare_gallery(model, candidates_or_gallery)
    chosen = None
    if candidate_lists is not None:
        chosen = []
        for q in qids:
            row = candidate_lists[q]
            if isinstance(row, (str, bytes)) or len(row) != len(set(row)):
                raise ValueError(f"Expected distinct image IDs in pool for {q!r}.")
            try:
                chosen.append([gallery.index[i] for i in row])
            except KeyError as exc:
                raise ValueError(f"Unknown image ID in pool for {q!r}: {exc.args[0]!r}") from exc
    if not gallery.ids or (chosen is not None and not any(chosen)):
        return empty
    pools, cosines = _get_pools(model, texts, gallery, width, chosen)
    tokens = _blip_text_tokens(model["blip_processor"], texts, model["device"])
    margins, unique, compact = _pair_margins(model, gallery, pools, tokens, pair_enabled)
    orders, fused, spans = [], [], []
    start = 0
    for pool in pools:
        span = slice(start, start + len(pool))
        spans.append(span)
        if len(mix) == 1:
            score = margins[next(iter(mix))][span]
        else:
            score = np.zeros(len(pool), dtype=np.float64)
            if pool:
                for arm, w in mix.items():
                    values = margins[arm][span].astype(np.float64)
                    score += w * (values - values.mean()) / max(float(values.std()), 1e-8)
        fused.append(score)
        orders.append(np.argsort(-score, kind="stable"))
        start += len(pool)
    evidence = _apply_pair(model, orders, pools, tokens, unique, compact) if pair_enabled else {}
    output = {}
    for i, qid in enumerate(qids):
        output[qid] = [
            {"image_id": gallery.ids[pools[i][col]], "siglip_cosine": float(cosines[i][col]),
             "margins": {arm: float(scores[spans[i].start + col]) for arm, scores in margins.items()},
             "score": float(fused[i][col]), "rank": rank}
            for rank, col in enumerate(orders[i], start=1)
        ]
        if i in evidence:
            for col, row in zip(orders[i][:2], output[qid][:2]):
                row["rank_before_pair"], row["pair_win_probability"] = evidence[i][col]
    return output


@torch.inference_mode()
def predict(model, queries, candidates, top_k=5):
    """Starterkit entry point. Return a ranked list of distinct image IDs per prompt."""
    top_k = _width(top_k)
    if top_k == 0:
        ids, _ = _query_fields(queries)
        return {q: [] for q in ids}
    rows = score_candidates(model, queries, candidates, width=max(_width(model["width"]), top_k))
    return {q: [r["image_id"] for r in candidates[:top_k]] for q, candidates in rows.items()}


