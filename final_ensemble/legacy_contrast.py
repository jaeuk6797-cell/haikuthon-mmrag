"""Optional team-side variants from HAIKUTHON_Spatial_Color_Colab.ipynb.

Example (run from the restored weight directory with the team scripts on sys.path):

    from model import load_model, prepare_gallery
    from legacy_contrast import legacy_scores
    model = load_model(".", "cpu", arms=["A"], use_pair=False)
    gallery = prepare_gallery(model, candidates)
    variants = legacy_scores(model, queries, candidates, gallery=gallery)
    score_rows = variants["spatial_color_top5"]  # prompt_id -> ranked score rows

These variants score only the original SigLIP five for each query. For score fusion,
score those same candidate IDs with the partner model. The rankblend variant uses
public BLIP A; requesting only the cue-based methods needs no BLIP forward. The
included SigLIP has the final group's retriever heads, so the historical validation
scores from the early retriever do not carry over automatically. Reflected candidates
need an additional mirrored SigLIP pass over the original PNG, once per unique
eligible candidate. The official inference budget for these optional methods is unmeasured.
"""

from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps

from model import _blip_text_tokens, _features, _get_pools, _pair_margins, _query_fields, prepare_gallery

import re

# Only screen-relative left/right followed by "of". Refuse to edit prompts that also mention
# a hand, a camera side, or any other left/right expression outside this grammar.
_DIRECTION = re.compile(r"\b(left|right)(?=\s+of\b)", re.I)
_ALL_DIRECTION = re.compile(r"\b(?:left|right)\b", re.I)

def _keep_case(source, replacement):
    if source.isupper(): return replacement.upper()
    if source[0].isupper(): return replacement.capitalize()
    return replacement

def reflected_prompt(text):
    if _DIRECTION.search(text) is None or len(_DIRECTION.findall(text)) != len(_ALL_DIRECTION.findall(text)):
        return None
    return _DIRECTION.sub(lambda m: _keep_case(m[0], "right" if m[0].lower()=="left" else "left"), text)

# This deliberately limited grammar accepts known object nouns adjacent to color adjectives.
# Keeping it fixed lets us measure its coverage and prevents ambiguous "orange" (the fruit),
# color words describing lighting, or an undifferentiated pair of same-kind objects from being swapped.
_COLOR = re.compile(r"\b(?:red|orange|yellow|green|blue|purple|pink|brown|black|white|gr[ae]y|beige)\b", re.I)
_CUE_OBJECTS = """
helmet|single ski|ski|skis|ski boot|scarf|umbrella|watering can|armchair|bowl|tennis ball|ball|surfboard|snowboard|
baseball bat|acoustic guitar|electric guitar|guitar|chair|bench|table|desk|cabinet|shelf|box|basket|suitcase|bag|
backpack|shirt|coat|single sock|sock|single glove|glove|kite|vase|pitcher|teapot|pot|saucepan|bottle|cup|mug|
pencil|pen|eraser|notebook|book|laptop|tablet device|camera|robot toy|toy car|teddy bear|picture frame|frame|
cone|cube|pyramid|wristwatch|watch|clock|tape measure|ruler|paintbrush|paint roller|gardening trowel|hand saw|knife|
fork|spoon|pliers|car|truck|bicycle|motorcycle|elephant|chicken|parrot|owl|bird|rabbit|cat|dog|hamster|
giraffe|guinea pig|apple|banana|donut|sandwich|loaf of bread|pumpkin|cushion|sofa|sleeping bag|tent|dress|
wall clock|tablet|stapler|rug|hanger|roller skate|clipboard|dumbbell|skateboard|frying pan|sleeping bag
"""
_CUE_NOUN = "(?:"+"|".join(re.escape(v) for v in sorted({v.strip() for v in _CUE_OBJECTS.replace("\n","").split("|") if v.strip()},key=lambda s:(-len(s),s)))+")"
_CUE_MOD = r"(?:(?:solid-colored|polka-dotted|striped|checkered|floral|round|square|oval|rectangular|hexagonal|plastic|metal|wooden|wood|ceramic|glass|single|open|closed)\s+){0,4}"
_AFTER_COLOR = re.compile(r"^\s+"+_CUE_MOD+"("+_CUE_NOUN+r")\b", re.I)
_BEFORE_COLOR = re.compile(r"\b("+_CUE_NOUN+r")\s*(?:,\s*(?:which\s+is\s+)?|(?:that\s+is|is)\s+)$", re.I)
_ROLE = re.compile(r"\b(wears?|wearing|holds?|holding)\s+(?:(?:one|the|a|an)\s+)?$",re.I)

def _cue_color(v):
    return v.lower().replace("grey","gray")

def swapped_color_prompt(text):
    occurrences=list(_COLOR.finditer(text))
    colors=sorted({_cue_color(m[0]) for m in occurrences})
    if len(colors)!=2: return None
    anchors={c:set() for c in colors}
    roles={c:set() for c in colors}
    for m in occurrences:
        prefix=text[:m.start()]
        following=_AFTER_COLOR.search(text[m.end():])
        previous=_BEFORE_COLOR.search(prefix)
        if following:
            noun=following[1].lower()
            leader=prefix
        elif previous and m[0].lower()!="orange":  # reject an unresolved fruit/attribute reading
            noun=previous[1].lower()
            leader=prefix[:previous.start(1)]
        else:
            return None
        anchors[_cue_color(m[0])].add(noun)
        role=_ROLE.search(leader)
        if role:
            roles[_cue_color(m[0])].add("wear" if role[1].lower().startswith("wear") else "hold")
    a,b=colors
    if not (len(anchors[a])==len(anchors[b])==1): return None
    if anchors[a]==anchors[b] and not ({frozenset(roles[a]),frozenset(roles[b])}=={frozenset(["hold"]),frozenset(["wear"])}):
        return None
    mapping={a:b,b:a}
    return _COLOR.sub(lambda m:_keep_case(m[0],mapping[_cue_color(m[0])]),text)


METHODS = ("spatial_flip_top5", "color_swap_top5", "spatial_color_top5", "blip_rankblend_top5")


def _edited_vectors(net, texts, edit):
    processor, device = net["siglip_processor"], net["device"]
    proposed = [(i, changed) for i, text in enumerate(texts) if (changed := edit(text))]
    result = {}
    for start in range(0, len(proposed), 64):
        block = proposed[start:start + 64]
        original, changed = ([texts[i] for i, _ in block], [s for _, s in block])
        def tokenize(words):
            return processor(text=words, padding="max_length", truncation=True, return_tensors="pt")
        base = tokenize(original)["input_ids"]
        tokenized = tokenize(changed)
        visible = (base != tokenized["input_ids"]).any(dim=1)
        if not visible.any():
            continue
        tokens = {key: tokenized[key][visible].to(device)
                  for key in ("input_ids", "attention_mask") if key in tokenized}
        values = _features(net["siglip"].get_text_features(**tokens)).cpu().numpy()
        result.update((block[i][0], value) for i, value in zip(visible.nonzero().flatten().tolist(), values))
    return result


def _mirror_vectors(net, positions, gallery, by_id):
    vectors = {}
    for start in range(0, len(positions), 32):
        block = positions[start:start + 32]
        photos = []
        try:
            for index in block:
                with Image.open(by_id[gallery.ids[index]]) as image, image.convert("RGB") as rgb:
                    photos.append(ImageOps.mirror(rgb))
            pixels = net["siglip_processor"](images=photos, return_tensors="pt")["pixel_values"]
            pixels = pixels.to(device=net["device"], dtype=net["dtype"])
            values = _features(net["siglip"].get_image_features(pixel_values=pixels)).cpu().numpy()
            vectors.update(zip(block, values))
        finally:
            for photo in photos:
                photo.close()
    return vectors


def _variant_scores(base, spatial, color, blip, methods):
    available = {
        "spatial_flip_top5": base + spatial,
        "color_swap_top5": base + color,
        "spatial_color_top5": base + (spatial + color),
    }
    if blip is not None:
        # Negative rank sum gives larger values to better rows; ties retain SigLIP order.
        order = np.argsort(-blip, kind="stable")
        available["blip_rankblend_top5"] = -(np.arange(len(base)) + np.argsort(order))
    return {name: available[name] for name in methods}


@torch.inference_mode()
def legacy_scores(net, queries, candidates, *, gallery=None, methods=METHODS):
    """Return {method: {prompt_id: [{image_id, score, rank}, ...]}} for the SigLIP five.

    candidates is always the official list of image ID/path dictionaries. A gallery
    from prepare_gallery(net, candidates) avoids another full decode and retrieval
    embedding pass. Request any subset of METHODS with methods=("color_swap_top5",).
    """
    methods = (methods,) if isinstance(methods, str) else tuple(methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError("Expected distinct legacy methods from " + ", ".join(METHODS))
    blend = "blip_rankblend_top5" in methods
    if blend and "A" not in net["heads"]:
        raise ValueError("Load the public BLIP with arms=['A'] or include A alongside other arms.")
    qids, texts = _query_fields(queries)
    output = {method: {q: [] for q in qids} for method in methods}
    if not qids or not candidates:
        return output
    gallery = prepare_gallery(net, candidates if gallery is None else gallery)
    by_id = {item["image_id"]: Path(item["image_path"]) for item in candidates}
    if len(by_id) != len(candidates) or set(by_id) != set(gallery.ids):
        raise ValueError("The input list must contain precisely the gallery image IDs without duplicates.")
    pools, raw = _get_pools(net, texts, gallery, 5, None)
    base = np.asarray(raw, dtype=np.float32)
    spatial, color = np.zeros_like(base), np.zeros_like(base)

    if {"spatial_flip_top5", "spatial_color_top5"}.intersection(methods):
        reflected = _edited_vectors(net, texts, reflected_prompt)
        used = sorted({i for row in reflected for i in pools[row]})
        mirror = _mirror_vectors(net, used, gallery, by_id)
        for row, text_vector in reflected.items():
            sim = np.stack([mirror[i] for i in pools[row]]) @ text_vector
            spatial[row] = 0.5 * (sim - base[row])

    if {"color_swap_top5", "spatial_color_top5"}.intersection(methods):
        swapped = _edited_vectors(net, texts, swapped_color_prompt)
        # Transfer only small, requested blocks from the existing gallery.
        rows = list(swapped)
        for start in range(0, len(rows), 64):
            block = rows[start:start + 64]
            idx = torch.tensor([pools[row] for row in block], device=gallery.vectors.device)
            image_vectors = gallery.vectors[idx].cpu().numpy()
            for row, images in zip(block, image_vectors):
                color[row] = 0.5 * (base[row] - images @ swapped[row])

    margins = None
    if blend:
        tokens = _blip_text_tokens(net["blip_processor"], texts, net["device"])
        public = dict(net, heads={"A": net["heads"]["A"]})
        components, _, _ = _pair_margins(public, gallery, pools, tokens, False)
        margins = components["A"].reshape(base.shape)
    if not all(np.isfinite(x).all() for x in (base, spatial, color)):
        raise FloatingPointError("Nonfinite contrast scores.")
    for row, qid in enumerate(qids):
        scores = _variant_scores(base[row], spatial[row], color[row],
                                  None if margins is None else margins[row], methods)
        for method, values in scores.items():
            output[method][qid] = [{"image_id": gallery.ids[pools[row][i]], "score": float(values[i]),
                                    "rank": rank} for rank, i in enumerate(np.argsort(-values, kind="stable"), 1)]
    return output
