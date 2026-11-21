"""Score a common candidate pool, or combine two already-scored pools by exact IDs."""
import argparse
import json
import math
from pathlib import Path


def z_scores(values):
    if not values:
        return []
    if not all(math.isfinite(v) for v in values):
        raise ValueError("The input contains a nonfinite score.")
    mean = sum(values) / len(values)
    std = max((sum((v - mean)**2 for v in values) / len(values))**0.5, 1e-8)
    return [(v - mean) / std for v in values]


def score_index(rows):
    by_id = {}
    for row in rows:
        image_id = row["image_id"]
        if not isinstance(image_id, str) or image_id in by_id:
            raise ValueError("Expected distinct image ID strings.")
        by_id[image_id] = float(row["score"])
    return by_id


def combine(ours, teammate, our_weight=0.5, top_k=5):
    if set(ours) != set(teammate) or not 0 <= our_weight <= 1 or top_k <= 0:
        raise ValueError("Both scorers need the same prompt IDs, a valid weight and a positive top_k.")
    ranked = {}
    for q, rows in ours.items():
        left, right = score_index(rows), score_index(teammate[q])
        if set(left) != set(right) or len(left) < top_k:
            raise ValueError(f"Both scorers must cover the same candidates, at least {top_k}, for {q!r}.")
        ids = list(left)
        a, b = z_scores(list(left.values())), z_scores([right[i] for i in ids])
        score = [our_weight*x + (1-our_weight)*y for x,y in zip(a,b)]
        indices = sorted(range(len(ids)), key=lambda i:-score[i])
        ranked[q] = [ids[i] for i in indices[:top_k]]
    return ranked


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score")
    score.add_argument("--model", type=Path, default=Path(__file__).resolve().parent)
    score.add_argument("--queries", required=True)
    score.add_argument("--candidates", required=True)
    score.add_argument("--pool", help="Optional {prompt_id:[image_ids]} shared by both scorers.")
    score.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    score.add_argument("--width", type=int, default=8)
    score.add_argument("--arms", help="Comma-separated components, e.g. S0,S1_noaux or S0.")
    score.add_argument("--pair", action="store_true", default=None, help="Apply locked Top-2 ordering.")
    score.add_argument("--out", required=True)
    mix = commands.add_parser("combine")
    mix.add_argument("--ours", required=True)
    mix.add_argument("--teammate", required=True)
    mix.add_argument("--our-weight", type=float, default=0.5)
    mix.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "combine":
        result = combine(read(args.ours), read(args.teammate), args.our_weight)
    else:
        from model import load_model, score_candidates
        options = {} if args.arms is None else {"arms":tuple(args.arms.split(","))}
        if args.pair is not None:
            options["use_pair"] = args.pair
        net = load_model(str(args.model), args.device, **options)
        result = score_candidates(net, read(args.queries), read(args.candidates),
                                  read(args.pool) if args.pool else None, width=args.width)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf8")


if __name__ == "__main__":
    main()
