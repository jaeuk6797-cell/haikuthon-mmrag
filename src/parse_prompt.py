"""Rule-based prompt parser: objects, attributes, spatial/action relations.

No LLM, no external API. Pure regex + vocabulary built from the train prompts.

parse_prompt(text) -> {
  "scene":     {"style", "view", "lighting", "location", "plain_backdrop"},
  "objects":   [{"id", "label", "head", "count", "attrs": {category: [values]}, "implicit"}],
  "relations": [{"type": "spatial"|"action", "pred", "subj", "obj", "text"}],
  "unresolved": [relation phrases whose subject/object could not be found],
  "isolated":  [ids of objects that appear in no relation],
}
Relations are directed: {"pred": "left_of", "subj": A, "obj": B} means "A is left of B".

CLI: python src/parse_prompt.py path/to/prompts.jsonl [--show N]
"""
import json
import random
import re
import sys
import collections as C

# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------
NOUNS = set(
    "person box cabinet bowl bench table desk scarf shelf ball helmet motorcycle suitcase tree "
    "train horse flower pot parrot stool cube pen bear violin frame bed car cylinder shirt "
    "pitcher tomato armchair cookie screwdriver pizza sofa skateboard glass chair tray rose pig "
    "tape slice egg bicycle bow scooter pumpkin mug strawberry eraser vase shovel apple apron "
    "corn rabbit trowel cupcake basketball glove umbrella bonsai robot cat snowboard kite "
    "saucepan knife lemon ear cheese handbag clock backpack ski boot guitar sneaker surfboard "
    "hat donut book notebook measure dog owl bat camera headphones cone paintbrush tent bird "
    "wedge colander sandwich cushion smartphone sunflower can saw pliers racket pear wristwatch "
    "coat tablet spoon envelope bread kettle dress fork broccoli wrench stapler turtle rake "
    "chicken loaf laptop donkey hamster scissors clipboard fern computer rack sunglasses "
    "dumbbell plate jacket penguin wheelbarrow pyramid tulip sheep cow ruler bag fox bottle "
    "banana bus goat carrot truck duck rug cactus deer cucumber pan croissant sock airplane "
    "elephant lamp boat helicopter hammer cart zebra teapot giraffe food container block skate "
    "orange pencil case roller".split()
)
IRR = {"scarves": "scarf", "shelves": "shelf", "cacti": "cactus", "loaves": "loaf",
       "strawberries": "strawberry", "tomatoes": "tomato", "people": "person", "knives": "knife",
       "foxes": "fox", "buses": "bus", "boxes": "box", "glasses": "glass", "sandwiches": "sandwich",
       "dresses": "dress", "skis": "ski", "pliers": "pliers", "scissors": "scissors",
       "headphones": "headphones", "sunglasses": "sunglasses"}
PARTITIVE = {"loaf", "wedge", "slice", "ear"}  # "loaf of bread", "ear of corn"

COLORS = set("brown white black yellow green red beige gray blue purple pink".split())  # orange handled below
PATTERN = {"solid-colored", "solid", "striped", "polka-dotted", "checkered", "floral",
           "zigzag-patterned", "patterned"}
MATERIAL = set("metal plastic wooden wood ceramic wicker leather rubber fabric cardboard stone".split())
SHAPE = set("star-shaped square rectangular round oval triangular hexagonal".split())
STATE = set("open closed folded unfolded inflated lit unlit snow-covered".split())
POSE = set("standing sitting lying kneeling crouching sleeping".split())
CATS = [("color", COLORS), ("pattern", PATTERN), ("material", MATERIAL),
        ("shape", SHAPE), ("state", STATE), ("pose", POSE)]
ATTR = COLORS | PATTERN | MATERIAL | SHAPE | STATE | POSE | {"orange"}
SKIP = {"is", "are", "that", "which", "made", "of", "and", ",", "also", "who", "has", "have", "solid", "device", "toy"}
QUANT = {"one": 1, "two": 2, "three": 3, "single": 1, "another": 1}
STOP = {"one", "two", "three", "another", "the", "a", "an", "is", "are", "with", "which", "who", "that", "where", "while"}

STYLES = [("photograph", r"photograph"), ("clay-style 3D render", r"clay-style"),
          ("low-poly 3D render", r"low-poly"), ("3D render", r"3d render"),
          ("watercolor", r"watercolor"), ("gouache", r"gouache"), ("oil painting", r"oil painting"),
          ("colored-pencil drawing", r"colored-pencil"), ("ink drawing", r"\bink\b"),
          ("paper-cut collage", r"paper-cut"), ("flat illustration", r"flat"),
          ("pencil sketch", r"pencil sketch")]
VIEWS = [("overhead", r"overhead"), ("front", r"front[- ]view|front view|from the front|front angle|from front|the front\b"),
         ("oblique", r"oblique"), ("side", r"side[- ]view|side view|from the side|side angle|in side view|the side\b")]
LIGHTS = [("side lighting", r"side lighting|side-lit|lit from the side"), ("warm indoor", r"warm indoor"),
          ("soft diffuse", r"soft diffuse"), ("direct sunlight", r"direct sunlight")]
LOCATIONS = ("studio|farm|park|living room|garden|bedroom|kitchen|office|patio|meadow|street|airport apron|"
             "airport|dining room|pasture|parking lot|classroom|forest|cafe|playground|beach|garage|"
             "orchard|lakeside|store|library|schoolyard|harbor|hall|workshop|campsite|field|home")
LOC_RE = re.compile(r"\b(?:in|on|at|inside)\s+(?:a|an|one|the)\s+(?:(?:[a-z\-]+\s){0,2})?(" + LOCATIONS + r")\b")
LOC_ANY_RE = re.compile(r"\b(?:a|an|one|the)\s+(?:(?:[a-z\-]+\s){0,2})?(" + LOCATIONS + r")\b")
LIGHT_SPAN_RE = re.compile(r"(?:under|in|with)\s+(?:soft|warm|direct|side)[a-z ]*?(?:lighting|sunlight)")
BACKDROP_RE = re.compile(r"studio with (?:a|one) plain backdrop|plain backdrop")

# relation phrases: (pred, regex, type, needs_mention_after)
REL_DEFS = [
    ("left_of", r"(?:to the )?left of", "spatial", False),
    ("right_of", r"(?:to the )?right of", "spatial", False),
    ("above", r"\babove\b", "spatial", False),
    ("below", r"\bbelow\b|\bbeneath\b", "spatial", False),
    ("next_to", r"next to|\bbeside\b|alongside|\bnear\b", "spatial", False),
    ("on", r"\bresting on\b|\brests on\b|\bsits on\b|\bon top of\b", "spatial", False),
    ("supports", r"\bsupports?\b", "spatial", False),
    ("contains", r"\bcontains?\b|\bcontaining\b", "spatial", False),
    ("on", r"\bon(?= which)", "spatial", False),
    ("inside", r"\binside\b", "spatial", False),
    ("in_front_of", r"in front of", "spatial", False),
    ("behind", r"\bbehind\b", "spatial", False),
    ("holds", r"\bholds?\b|\bholding\b", "action", False),
    ("wears", r"\bwears?\b|\bwearing\b", "action", False),
    ("rides", r"\brides?\b|\briding\b", "action", False),
    ("dir_left", r"to (?:its|their) left", "spatial", False),
    ("dir_right", r"to (?:its|their) right", "spatial", False),
    ("on", r"\bon\b", "spatial", True),
    ("inside", r"\bin\b", "spatial", True),
    ("below", r"\bunder\b", "spatial", True),
    ("above", r"\bover\b", "spatial", True),
]
REL_RES = [(p, re.compile(r), t, n) for p, r, t, n in REL_DEFS]


ORANGE_NEXT = {",", ".", "is", "are", "and", "which", "with", "that", "who", "above", "below", "beneath",
               "inside", "behind", "next", "to", "on", "in", "under", "over", "beside", "near", "rests",
               "sits", "resting", "holds", "holding", "appears", "appear", "stands", "lies", "alongside"}


def norm(w):
    if w in IRR:
        return IRR[w]
    if w in NOUNS:
        return w
    if w.endswith("s") and w[:-1] in NOUNS:
        return w[:-1]
    if w.endswith("es") and w[:-2] in NOUNS:
        return w[:-2]
    return None


def attr_cat(w):
    if w == "orange":
        return "color"
    for name, s in CATS:
        if w in s:
            return name


def tokenize(s):
    return [(m.group(0), m.start()) for m in re.finditer(r"[a-z]+(?:-[a-z]+)*|[,.]", s)]


# --------------------------------------------------------------------------
# mentions
# --------------------------------------------------------------------------
def is_noun_at(toks, k):
    """Noun at k; 'orange' is a noun only right after a quantity word (the fruit)."""
    w = toks[k][0]
    if w == "orange":
        if k == 0 or toks[k - 1][0] not in QUANT and toks[k - 1][0] not in ("the", "a", "an", "another"):
            return False
        nxt = toks[k + 1][0] if k + 1 < len(toks) else "."
        return nxt in ORANGE_NEXT
    if w == "apron" and k > 0 and toks[k - 1][0] == "airport":
        return False
    if w == "train" and k + 1 < len(toks) and toks[k + 1][0] == "platform":
        return False
    return norm(w) is not None


def find_mentions(toks):
    words = [t[0] for t in toks]
    n = len(toks)
    out = []
    i = 0
    while i < n:
        w = words[i]
        if w not in QUANT and w not in ("the", "a", "an"):
            i += 1
            continue
        is_the = w in ("the", "a", "an")
        limit = i + (5 if is_the else 8)
        j = None
        k = i + 1
        while k < min(limit, n):
            t = words[k]
            if t == ".":
                break
            if t in STOP and k > i + 1:
                break
            if t == "of" and words[k - 1] not in ("pair", "pairs"):
                break
            if (t not in ATTR or t == "orange") and t not in ("and", ",", "of", "pair", "pairs", "small") and is_noun_at(toks, k):
                if t == "pencil" and k + 1 < n and words[k + 1] == "sketch":
                    break
                j = k
                break
            k += 1
        if j is None:
            i += 1
            continue
        head = j
        compound = [words[j]]
        # consecutive nouns: "table knife", "tape measure", "bonsai tree" -> last one is the head
        while head + 1 < n and is_noun_at(toks, head + 1) and words[head + 1] not in ATTR:
            head += 1
            compound.append(words[head])
        label_head = norm(words[head]) or words[head]
        if words[head] in PARTITIVE and head + 2 < n and words[head + 1] == "of" and is_noun_at(toks, head + 2):
            head += 2
            label_head = norm(words[head]) or words[head]
        pre_words = words[i + 1:j]
        mods = [x for x in pre_words if x not in ATTR and x not in ("and", ",", "of", "pair", "pairs") and x not in QUANT]
        pre = [x for x in pre_words if x in ATTR]
        if label_head == "bag" and "sleeping" in pre:
            pre.remove("sleeping")
            mods.append("sleeping")
        compound_nouns = [norm(x) or x for x in compound[:-1]] if len(compound) > 1 else []
        label = " ".join(mods + compound_nouns + [label_head])
        # post attributes: "..., which is open, blue and polka-dotted", "made of pink metal"
        post = []
        k = head + 1
        while k < n and (words[k] in ATTR or words[k] in SKIP):
            if words[k] in ATTR:
                post.append(words[k])
            k += 1
        count = 1
        if words[i] in ("two", "three"):
            count = QUANT[words[i]]
        elif is_the and any(x in words[i + 1:j] for x in ("two", "three")):
            count = 2
        # "one pair of scissors" is a single object; "two pairs of sunglasses" are two

        post_direct = head + 1 < n and words[head + 1] in ("is", "are") and bool(post)
        out.append({"i0": i, "i1": head, "det": w, "ref": is_the, "label": label, "head": label_head,
                    "count": count, "attrs": pre + post, "pre": pre, "post": post,
                    "post_direct_is": post_direct, "mods": mods})
        i = head + 1
    return out


# --------------------------------------------------------------------------
# relations
# --------------------------------------------------------------------------
def scene_spans(s):
    spans = [m.span() for m in LOC_RE.finditer(s)]
    spans += [m.span() for m in LIGHT_SPAN_RE.finditer(s)]
    spans += [m.span() for m in BACKDROP_RE.finditer(s)]
    return spans


def find_relations(s, toks, mentions):
    spans = scene_spans(s)
    starts = [t[1] for t in toks]
    mention_start_chars = [starts[m["i0"]] for m in mentions]
    cands = []
    for pred, rx, typ, needs in REL_RES:
        for m in rx.finditer(s):
            a, b = m.span()
            if any(a < e and b > st for st, e in spans):
                continue
            if needs:
                tail = s[b:b + 3]
                if not any(0 <= ms - b <= 3 for ms in mention_start_chars):
                    continue
                if re.match(r"\s*(?:the\s+)?(?:left|right)\b", s[b:b + 12]):
                    continue
            cands.append((a, b, pred, typ))
    cands.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    chosen = []
    last_end = -1
    for a, b, pred, typ in cands:
        if a < last_end:
            continue
        chosen.append((a, b, pred, typ))
        last_end = b
    rels = []
    for a, b, pred, typ in chosen:
        r0 = next((k for k, st in enumerate(starts) if st >= a), len(toks))
        r1 = next((k for k, st in enumerate(starts) if st >= b), len(toks))
        rels.append({"pred": pred, "type": typ, "r0": r0, "r1": r1, "text": s[a:b]})
    return rels


def link_relations(s, toks, mentions, rels):
    """Fill subj/obj (as mention indices) for each relation."""
    words = [t[0] for t in toks]
    results = []
    prev_rel = None
    for r in rels:
        pred, r0, r1 = r["pred"], r["r0"], r["r1"]
        before = [k for k, m in enumerate(mentions) if m["i1"] < r0]
        after = [k for k, m in enumerate(mentions) if m["i0"] >= r1]
        subj = obj = None
        nxt_word = words[r1] if r1 < len(words) else ""
        inverted_after_and = (before and r0 >= 1 and words[r0 - 1] == "and" and after
                              and any(words[k] in ("is", "are") and words[k - 1] not in ("which", "that", "who")
                                      for k in range(mentions[after[0]]["i1"] + 1, len(words))))
        if nxt_word in ("it", "them") and before:
            subj = before[-1]
            earlier = before[:-1]
            obj = earlier[-1] if earlier else None
        elif pred in ("dir_left", "dir_right"):
            if before:
                subj = before[-1]
                earlier = [k for k in before[:-1] if mentions[k]["label"] != mentions[subj]["label"] or True]
                obj = earlier[-1] if earlier else None
            pred = "left_of" if pred == "dir_left" else "right_of"
        elif nxt_word in ("which", "whom") and before:
            obj = before[-1]
            aft = [k for k in after if mentions[k]["i0"] > r1]
            subj = aft[0] if aft else None
        elif (not before or inverted_after_and) and after:
            obj = after[0]
            subj = None
            k_is = None
            for k in range(mentions[obj]["i1"] + 1, len(words)):
                if words[k] in ("is", "are", "rests", "sits", "stands", "lies", "appears", "appear", "sit", "stand") and words[k - 1] not in ("which", "that", "who"):
                    k_is = k
                    break
            if k_is is not None:
                aft = [k for k in after if mentions[k]["i0"] > k_is and k != obj]
                subj = aft[0] if aft else None
            if subj is None:
                comma_after = [k for k in after if k != obj and mentions[k]["i0"] > 0 and words[mentions[k]["i0"] - 1] == ","]
                if comma_after:
                    subj = comma_after[-1]
        else:
            subj = before[-1] if before else None
            obj = after[0] if after else None
            if (r0 >= 2 and words[r0 - 1] in ("is", "are") and words[r0 - 2] == ","):
                # "A, which rests on B, is to the left of C": subject is A, not B
                wh = [k for k in range(r0) if words[k] in ("which", "who", "that")]
                if wh:
                    head_cands = [k for k, m in enumerate(mentions) if m["i1"] < wh[-1]]
                    if head_cands:
                        subj = head_cands[-1]
            if (subj is not None and prev_rel is not None and prev_rel.get("obj") == subj
                    and prev_rel.get("subj") is not None):
                between = words[mentions[subj]["i1"] + 1:r0]
                if all(x in ("and", ",", "is", "are") for x in between) and "and" in between:
                    subj = prev_rel["subj"]
        if pred == "supports":
            subj, obj = obj, subj
            pred = "on"
        if pred == "contains":
            if not before:
                continue  # "A studio ... contains one X": scene, not an object relation
            subj, obj = obj, subj
            pred = "inside"
        r["subj"], r["obj"], r["pred"] = subj, obj, pred
        results.append(r)
        if subj is not None and obj is not None:
            prev_rel = r
    return results


# --------------------------------------------------------------------------
# parse
# --------------------------------------------------------------------------
def first_match(lst, text):
    for name, rx in lst:
        if re.search(rx, text):
            return name
    return None


def parse_scene(text):
    low = text.lower()
    loc = LOC_RE.search(low) or LOC_ANY_RE.search(low)
    return {
        "style": first_match(STYLES, low),
        "view": first_match(VIEWS, low),
        "lighting": first_match(LIGHTS, low),
        "location": loc.group(1) if loc else None,
        "plain_backdrop": bool(BACKDROP_RE.search(low)),
    }


def parse_prompt(text):
    low = text.lower()
    objects = []  # list of dict
    relations = []
    unresolved = []
    sentences = [x for x in re.split(r"(?<=[.])\s+", low.strip()) if x]
    for sent_idx, sent in enumerate(sentences):
        in_first_sentence = sent_idx == 0
        toks = tokenize(sent)
        mentions = find_mentions(toks)
        if not mentions:
            continue
        rels = link_relations(sent, toks, mentions, find_relations(sent, toks, mentions))
        # "A resting on B is open": the plain 'is ATTR' after a relation's object describes the subject
        for r in rels:
            if r["subj"] is not None and r["obj"] is not None and r["subj"] != r["obj"]:
                mo, ms = mentions[r["obj"]], mentions[r["subj"]]
                if mo["post_direct_is"] and mo["post"] and mo["i0"] > ms["i1"]:
                    moved = mo["post"]
                    mo["attrs"] = [a for a in mo["attrs"] if a not in moved or a in mo["pre"]]
                    ms["attrs"] = ms["attrs"] + [a for a in moved if a not in ms["attrs"]]
                    mo["post"] = []
                    mo["post_direct_is"] = False
        # map mentions to object ids (introduce or resolve "the X")
        ids = []
        for m in mentions:
            oid = None
            if m["ref"]:
                cands = [o for o in objects if o["head"] == m["head"]]
                if cands:
                    ma = set(m["attrs"])
                    scored = sorted(cands, key=lambda o: -len(ma & {v for vs in o["attrs"].values() for v in vs}))
                    best = scored[0]
                    oid = best["id"]
                    for a in m["attrs"]:
                        cat = attr_cat(a)
                        if a not in best["attrs"].setdefault(cat, []):
                            best["attrs"][cat].append(a)
            if oid is None and not m["attrs"] and m["det"] == "one" and not in_first_sentence:
                cands = [o for o in objects if o["head"] == m["head"] and o["sent"] < sent_idx]
                grouped = [o for o in cands if o["count"] >= 2]
                pick = (grouped or cands or [None])[0]
                if pick is not None:
                    oid = pick["id"]
            if oid is None:
                oid = len(objects)
                attrs = {}
                for a in m["attrs"]:
                    attrs.setdefault(attr_cat(a), [])
                    if a not in attrs[attr_cat(a)]:
                        attrs[attr_cat(a)].append(a)
                objects.append({"id": oid, "label": m["label"], "head": m["head"], "count": m["count"],
                                "attrs": attrs, "implicit": m["ref"], "sent": sent_idx})
            ids.append(oid)
        for r in rels:
            if r["subj"] is None or r["obj"] is None or ids[r["subj"]] == ids[r["obj"]]:
                unresolved.append(r["text"])
            else:
                relations.append({"type": r["type"], "pred": r["pred"], "subj": ids[r["subj"]],
                                  "obj": ids[r["obj"]], "text": r["text"]})
    for o in objects:
        o.pop("sent", None)
    used = {r["subj"] for r in relations} | {r["obj"] for r in relations}
    return {"scene": parse_scene(text), "objects": objects, "relations": relations,
            "unresolved": unresolved, "isolated": [o["id"] for o in objects if o["id"] not in used]}


def describe(parsed):
    """Readable one-line-per-fact summary (used for eyeballing)."""
    lines = []
    for o in parsed["objects"]:
        a = ", ".join(v for vs in o["attrs"].values() for v in vs) or "-"
        lines.append("  obj%d %s x%d [%s]%s" % (o["id"], o["label"], o["count"], a, " (ref)" if o["implicit"] else ""))
    for r in parsed["relations"]:
        s = parsed["objects"][r["subj"]]["label"]
        o = parsed["objects"][r["obj"]]["label"]
        lines.append("  %s(obj%d %s -> obj%d %s)" % (r["pred"], r["subj"], s, r["obj"], o))
    for u in parsed["unresolved"]:
        lines.append("  UNRESOLVED: %s" % u)
    return "\n".join(lines)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "prompts.jsonl"
    show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 0
    P = [json.loads(l) for l in open(path, encoding="utf-8")]
    parsed = [parse_prompt(p["prompts"]) for p in P]
    n_obj = sum(len(p["objects"]) for p in parsed)
    n_rel = sum(len(p["relations"]) for p in parsed)
    n_unres = sum(len(p["unresolved"]) for p in parsed)
    n_iso = sum(len(p["isolated"]) for p in parsed)
    print("prompts %d | objects %d (%.2f/prompt) | relations %d | unresolved %d (%.1f%% of relation phrases) | isolated objects %d (%.1f%%)" % (
        len(P), n_obj, n_obj / len(P), n_rel, n_unres, 100 * n_unres / max(1, n_rel + n_unres), n_iso, 100 * n_iso / n_obj))
    print("preds", C.Counter(r["pred"] for p in parsed for r in p["relations"]).most_common())
    print("scene location", C.Counter(p["scene"]["location"] for p in parsed).most_common(6))
    for k in ("style", "view", "lighting"):
        print("scene", k, "missing:", sum(1 for p in parsed if p["scene"][k] is None))
    print("prompts with no object parsed:", sum(1 for p in parsed if not p["objects"]),
          "| with unresolved relation:", sum(1 for p in parsed if p["unresolved"]))
    random.seed(3)
    for idx in random.sample(range(len(P)), show):
        print("---", P[idx]["id"], P[idx]["prompts"])
        print(describe(parsed[idx]))
