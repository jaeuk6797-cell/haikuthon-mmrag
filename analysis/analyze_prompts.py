"""prompts.jsonl 구조 분석: 길이, 문장 구성, 개수/색/관계/스타일/시점/조명 분포, 중복 여부.

사용: python analyze_prompts.py <prompts.jsonl>
"""

import json
import re
import sys
from collections import Counter

path = sys.argv[1]
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
texts = [r["prompts"] for r in rows]
n = len(texts)


def pct(x):
    return f"{100 * x / n:.1f}%"


def top(counter, k=15):
    return ", ".join(f"{w}({c})" for w, c in counter.most_common(k))


print(f"== 기본 ==\n쿼리 수: {n}, 고유 프롬프트: {len(set(texts))}")
chars = sorted(len(t) for t in texts)
words = sorted(len(t.split()) for t in texts)
print(f"글자수 min/median/max: {chars[0]}/{chars[n // 2]}/{chars[-1]}")
print(f"단어수 min/median/max: {words[0]}/{words[n // 2]}/{words[-1]}")

sents = [re.split(r"(?<=\.)\s+", t.strip()) for t in texts]
print("문장 수 분포:", sorted(Counter(len(s) for s in sents).items()))

# 개수 표현
num_words = "one two three four five six seven eight nine ten".split()
cnt = Counter()
for t in texts:
    for w in re.findall(r"\b(" + "|".join(num_words) + r")\b", t.lower()):
        cnt[w] += 1
print("\n== 개수 표현 ==\n", top(cnt, 10))
only_one = sum(1 for t in texts if not re.search(r"\b(two|three|four|five|six|seven|eight|nine|ten)\b", t.lower()))
print(f"'one'만 쓰는 프롬프트: {only_one} ({pct(only_one)})")
n_obj = Counter(len(re.findall(r"\b[Oo]ne\b", t)) for t in texts)
print("프롬프트당 'One' 등장 횟수:", sorted(n_obj.items()))

# 색/무늬
colors = "red orange yellow green blue purple pink brown black white gray grey beige silver gold".split()
col = Counter()
for t in texts:
    for w in re.findall(r"\b(" + "|".join(colors) + r")\b", t.lower()):
        col[w] += 1
print("\n== 색 ==\n", top(col, 20))
col_per = Counter(len(re.findall(r"\b(" + "|".join(colors) + r")\b", t.lower())) for t in texts)
print("프롬프트당 색 단어 수:", sorted(col_per.items()))
pat = Counter()
for t in texts:
    for w in re.findall(r"\b(solid-colored|polka-dotted|striped|checkered|plaid|floral|patterned|spotted)\b", t.lower()):
        pat[w] += 1
print("무늬:", top(pat))

# 공간/동작 관계
rel_patterns = {
    "above": r"\babove\b", "below": r"\bbelow\b", "left of": r"to the left of", "right of": r"to the right of",
    "next to": r"\bnext to\b", "on/resting on": r"\b(resting on|rests on|on top of)\b", "wearing": r"\bwearing\b",
    "in front of": r"in front of", "behind": r"\bbehind\b", "holding": r"\bholding\b", "under/beneath": r"\b(under|beneath)\b",
}
print("\n== 관계 (포함하는 프롬프트 수) ==")
for k, rgx in rel_patterns.items():
    c = sum(1 for t in texts if re.search(rgx, t.lower()))
    print(f"{k}: {c} ({pct(c)})")
rel_count = Counter(
    sum(1 for rgx in rel_patterns.values() if re.search(rgx, t.lower())) for t in texts
)
print("프롬프트당 관계 종류 수:", sorted(rel_count.items()))

# 마지막 문장(스타일/시점/조명) 분해
style = Counter()
view = Counter()
light = Counter()
for t in texts:
    low = t.lower()
    m = re.search(r"(photograph|watercolor painting|colored-pencil drawing|clay-style 3d render|flat illustration|oil painting|pixel art|sketch|3d render|illustration|painting|drawing|render)", low)
    style[m.group(1) if m else "기타"] += 1
    m = re.search(r"(front view|side view|top view|top-down view|high-angle|low-angle|from the front|from the side|viewed from the (?:front|side|top)|overhead|aerial)", low)
    view[m.group(1) if m else "기타"] += 1
    m = re.search(r"(direct sunlight|soft diffuse lighting|warm indoor lighting|side lighting|backlighting|cool [a-z ]*lighting|[a-z]+ lighting|[a-z]+ light)", low)
    light[m.group(1) if m else "기타"] += 1
print("\n== 스타일 ==\n", top(style, 12))
print("== 시점 ==\n", top(view, 12))
print("== 조명 ==\n", top(light, 12))

# 장소
place = Counter()
for t in texts:
    m = re.search(r"\b(?:in|on|at)\s+(?:a|an|the)\s+([a-z\- ]+?)(?:[.,]| with| viewed| from)", t.lower())
    if m:
        place[m.group(1)] += 1
print("\n== 장소 후보 ==\n", top(place, 25), f"\n(장소 추출 성공 {sum(place.values())}/{n})")

# 객체 명사 후보: 'One <adj...> <noun>' 패턴의 마지막 단어
objs = Counter()
for t in texts:
    for m in re.finditer(r"\b[Oo]ne ((?:[a-z\-]+ )*?)([a-z]+)(?= (?:is|are|rests?|resting|stands?|sits?|lies|lying|has|in|on|above|below|to|next|and|,)|[.,])", t):
        objs[m.group(2)] += 1
print("\n== 객체 명사 후보 ==\n", top(objs, 40), f"\n(고유 {len(objs)}종)")

# 접두 문장 템플릿 다양성
first = Counter(" ".join(s[0].split()[:3]) for s in sents)
print("\n== 첫 문장 시작 3단어 ==\n", top(first, 12))
