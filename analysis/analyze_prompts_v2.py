"""Train prompt structure analysis v2 (text only).

Usage: python analysis/analyze_prompts_v2.py path/to/prompts.jsonl

Everything is regex/vocabulary based, so numbers are approximate:
- Objects are the phrases introduced with one/two/three/single + noun.
- "Attributes" = intrinsic properties of one object (color, pattern, material,
  shape, state, pose). "Relations" = links between objects, split into spatial
  (left of, above, inside, ...) and action (holds, wears, rides).
- Two-word nouns (bonsai tree, tape measure, roller skate) are only partly handled.
"""
import json
import re
import sys
import collections as C

PATH = sys.argv[1] if len(sys.argv) > 1 else "prompts.jsonl"
T = [json.loads(l)["prompts"] for l in open(PATH, encoding="utf-8")]
L = [t.lower() for t in T]
N_PROMPTS = len(T)


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def count(pats):
    r = C.Counter()
    for l in L:
        for k, p in pats.items():
            if re.search(p, l):
                r[k] += 1
    return r


# --------------------------------------------------------------------------
# 1. Vocabulary and sentence variety
# --------------------------------------------------------------------------
FUNC = set(
    "one a is the in with and lighting from scene of to view an on viewed d warm soft diffuse "
    "painting render direct sunlight shown two which right left front are oblique next under "
    "above below as present also that rests setting inside shows rendered holds made set "
    "contains single pair appears it there who image picture appear sits resting holding side "
    "overhead studio plain backdrop three at its by up this them each other these both having "
    "top has".split()
)
words = [w for l in L for w in re.findall(r"[a-z]+(?:-[a-z]+)*", l)]
wc = C.Counter(words)
print("== vocabulary ==")
print("prompts", N_PROMPTS, "| vocab", len(wc), "| words seen <=3 times", sum(1 for v in wc.values() if v <= 3))
skel = C.Counter(
    " ".join(w if w in FUNC else "X" for w in re.findall(r"[a-z]+(?:-[a-z]+)*|[,.]", l)) for l in L
)
print("distinct sentence skeletons", len(skel))

# --------------------------------------------------------------------------
# 2. Scene-level fields
# --------------------------------------------------------------------------
styles = {
    "photograph": r"photograph", "3D render": r"(?<!clay-style )(?<!low-poly )3d render",
    "clay-style 3D": r"clay-style", "low-poly 3D": r"low-poly", "watercolor": r"watercolor",
    "gouache": r"gouache", "oil painting": r"oil painting", "colored-pencil": r"colored-pencil",
    "ink": r"\bink\b", "paper-cut collage": r"paper-cut", "flat illustration": r"flat",
    "pencil sketch": r"pencil sketch",
}
views = {
    "overhead": r"overhead", "front": r"front[- ]view|front view|from the front|front angle|from front",
    "oblique": r"oblique", "side": r"side[- ]view|side view|from the side|side angle|in side view",
}
lights = {
    "side lighting": r"side lighting|side-lit|lit from the side", "warm indoor": r"warm indoor",
    "soft diffuse": r"soft diffuse", "direct sunlight": r"direct sunlight",
}
print("\n== scene fields ==")
for name, d in [("style", styles), ("view", views), ("lighting", lights)]:
    print(name, count(d).most_common())

# --------------------------------------------------------------------------
# 3. Object-level parsing
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
    "elephant lamp boat helicopter hammer cart zebra teapot giraffe food container block skate".split()
)
IRR = {
    "scarves": "scarf", "shelves": "shelf", "cacti": "cactus", "loaves": "loaf",
    "strawberries": "strawberry", "tomatoes": "tomato", "people": "person", "knives": "knife",
    "foxes": "fox", "buses": "bus", "boxes": "box", "glasses": "glass", "sandwiches": "sandwich",
    "dresses": "dress", "skis": "ski", "pliers": "pliers", "scissors": "scissors",
    "headphones": "headphones", "sunglasses": "sunglasses",
}
COLORS = set("brown white black yellow green red beige orange gray blue purple pink".split())
PATTERN = {"solid-colored": 1, "solid": 1, "striped": 1, "polka-dotted": 1, "checkered": 1,
           "floral": 1, "zigzag-patterned": 1, "patterned": 1}
MATERIAL = set("metal plastic wooden wood ceramic wicker leather rubber fabric cardboard stone".split())
SHAPE = set("star-shaped square rectangular round oval triangular hexagonal".split())
STATE = set("open closed folded unfolded inflated lit unlit snow-covered".split())
POSE = set("standing sitting lying kneeling crouching sleeping".split())
ATTR = COLORS | set(PATTERN) | MATERIAL | SHAPE | STATE | POSE
SKIP = {"is", "are", "that", "which", "made", "of", "and", ",", "also", "who", "has", "have", "solid"}


def attr_cat(w):
    for name, s in [("color", COLORS), ("pattern", PATTERN), ("material", MATERIAL),
                    ("shape", SHAPE), ("state", STATE), ("pose", POSE)]:
        if w in s:
            return name


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


def tok(t):
    return re.findall(r"[a-z]+(?:-[a-z]+)*|[,.]", t.lower())


def post_attrs(tk, k):
    out = []
    while k < len(tk) and (tk[k] in ATTR or tk[k] in SKIP):
        if tk[k] in ATTR:
            out.append(tk[k])
        k += 1
    return out


def pre_attrs(tk, j):
    out, k = [], j - 1
    while k >= 0 and (tk[k] in ATTR or tk[k] in (",", "and", "made", "of")):
        if tk[k] in ATTR:
            out.append(tk[k])
        k -= 1
    return out


SPATIAL = (r"next to|beside|alongside|\bnear\b|left of|right of|on the left|on the right|\babove\b|"
           r"\bbelow\b|beneath|under (?:one|the|two|three)|on top of|resting on|rests on|sits on|"
           r"\bon one\b|\bon the\b|inside|in front of|behind|\bover\b|among|along\b|"
           r"to (?:its|their|the) (?:left|right)|supports|\bon which\b|\bon top\b")
ACTION = r"\bholds?\b|holding|wearing|wears|\brides?\b|riding"
SCENE = (r"(?:in|on|at|inside) (?:a|an|one|the) (?:[a-z\- ]+? )?(?:studio with (?:a|one) plain backdrop|"
         r"studio|farm|park|living room|garden|bedroom|kitchen|office|patio|meadow|street|"
         r"airport(?: apron)?|dining room|pasture|parking lot|classroom|forest|cafe|playground|"
         r"beach|garage|orchard|lakeside|store|library|schoolyard|harbor|hall|workshop|campsite|field)")


def clean(s):
    """Remove scene location and lighting phrases so they are not mistaken for relations."""
    s = re.sub(SCENE, "", s.lower())
    s = re.sub(r"studio with (?:a|one) plain backdrop", "", s)
    return re.sub(r"(?:under|in|with) (?:soft|warm|direct|side)[a-z ]*(?:lighting|sunlight)", "", s)


objs = []  # (prompt idx, noun, strict attrs, lenient attrs, in spatial sentence, in action sentence)
for pi, t in enumerate(T):
    tk = tok(t)
    n = len(tk)
    mention = C.defaultdict(set)
    for j, w in enumerate(tk):
        nn = norm(w)
        if nn:
            mention[nn] |= set(post_attrs(tk, j + 1)) | set(pre_attrs(tk, j))
    sents = [clean(s) for s in re.split(r"(?<=[.])\s+", t) if not re.match(r"(?i)(the scene|it is rendered)", s)]
    sp_s = [s for s in sents if re.search(SPATIAL, s)]
    ac_s = [s for s in sents if re.search(ACTION, s)]
    for i, w in enumerate(tk):
        if w not in ("one", "two", "three", "single"):
            continue
        j = None
        for k in range(i + 1, min(i + 7, n)):
            if tk[k] == ".":
                break
            if k > i + 1 and tk[k] in ("one", "two", "three", "is", "are", "with"):
                break
            if norm(tk[k]):
                j = k
                break
        if j is None:
            continue
        nn = norm(tk[j])
        strict = {x for x in tk[i + 1:j] if x in ATTR} | set(post_attrs(tk, j + 1))
        found = lambda L_: any(re.search(r"\b" + nn + r"(s|es)?\b", s) for s in L_)
        objs.append((pi, nn, strict, mention[nn] | strict, found(sp_s), found(ac_s)))

N = len(objs)
print("\n== objects ==")
print("objects", N, "| per prompt %.2f" % (N / N_PROMPTS))
for idx, lab in [(2, "strict"), (3, "lenient")]:
    cats = lambda a: {attr_cat(x) for x in a}
    print("[%s] no attribute %.1f%% | color %.1f%% | pattern %.1f%% | material %.1f%% | pose %.1f%%" % (
        lab, pct(sum(1 for o in objs if not o[idx]), N),
        *[pct(sum(1 for o in objs if c in cats(o[idx])), N) for c in ["color", "pattern", "material", "pose"]]))

print("\n== relation types, object level (spatial / action) ==")
for name, sel in [("all", lambda o: True), ("person", lambda o: o[1] == "person"),
                  ("non-person", lambda o: o[1] != "person")]:
    O = [o for o in objs if sel(o)]
    c = C.Counter((o[4], o[5]) for o in O)
    print("%-10s n=%-6d spatial only %.1f%% | action only %.1f%% | both %.1f%% | neither %.1f%%" % (
        (name, len(O)) + tuple(pct(c[k], len(O)) for k in [(True, False), (False, True), (True, True), (False, False)])))

print("\n== attribute x relation (lenient attrs) ==")
c = C.Counter((bool(o[3]), o[4] or o[5]) for o in objs)
print("attr+rel %.1f%% | attr only %.1f%% | rel only %.1f%% | neither %.1f%%" % tuple(
    pct(c[k], N) for k in [(True, True), (True, False), (False, True), (False, False)]))

isolated = [o for o in objs if not o[4] and not o[5]]
print("objects in no relation sentence:", len(isolated), "(%.1f%%)" % pct(len(isolated), N))
print("  of which no attribute either:", sum(1 for o in isolated if not o[3]))
print("  top nouns:", C.Counter(o[1] for o in isolated).most_common(12))

# --------------------------------------------------------------------------
# 4. Hard-negative proxy: prompts sharing the same object set
# --------------------------------------------------------------------------
sets = [frozenset(o[1] for o in objs if o[0] == pi) for pi in range(N_PROMPTS)]
by_set = C.Counter(sets)
print("\n== object sets ==")
print("distinct object sets", len(by_set), "| set-size distribution", sorted(C.Counter(by_set.values()).items())[:8])
inv = C.defaultdict(set)
for i, s in enumerate(sets):
    for x in s:
        inv[x].add(i)
import random
random.seed(0)
same, jac = [], []
for i in random.sample(range(N_PROMPTS), min(600, N_PROMPTS)):
    if not sets[i]:
        continue
    cand = set().union(*[inv[x] for x in sets[i]]) - {i}
    same.append(sum(1 for j in cand if sets[j] == sets[i]))
    jac.append(sum(1 for j in cand if len(sets[i] & sets[j]) / len(sets[i] | sets[j]) >= 0.5))
print("others with identical object set: mean %.1f max %d | Jaccard>=0.5: mean %.1f" % (
    sum(same) / len(same), max(same), sum(jac) / len(jac)))
print("(x4 for a 16,000 image pool if test is built the same way: ~%.0f identical, ~%.0f similar)" % (
    4 * sum(same) / len(same), 4 * sum(jac) / len(jac)))
