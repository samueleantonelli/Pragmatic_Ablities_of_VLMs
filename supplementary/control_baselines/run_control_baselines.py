"""
Control baselines for Referring Expression Comprehension with Qwen2.5-VL-7B.

Answers the reviewer's request: is near-ceiling, flat accuracy due to the description
content, or to visual salience / positional priors? For each (image, description) item
the listener is run under several conditions:

  real               the original description (reference condition)
  none               no description at all (prompt asks for "the target")
  shuffled_same_cat  description of a DIFFERENT target with the same category label
                     (and same prompt style, default/brief, to hold verbosity constant)
  shuffled_random    description of a different target of any category (same prompt style)
  category_only      (optional) just "a <label>"  -> how far category + salience gets you

The dataset is read in streaming mode:
  pass 1: text/metadata columns only (cheap), to build the shuffles and sample items
  pass 2: stream the images and run the model only on the sampled items

Parser notes:
  - Qwen2.5-VL answers with JSON ({"bbox_2d": [x1, y1, x2, y2], ...}). The box is read
    from the first [a, b, c, d] list in the answer.
  - Every record stores the size of the image seen by the model (rw, rh), a status
    (ok / no_box / invalid_box) and a flag for descriptions that still mention the red box.
  - Records written by an incompatible parser version are ignored when resuming.

Usage:
  python run_control_baselines.py --n-items 2000 --out results.jsonl
  python run_control_baselines.py --summarize-only --out results.jsonl
"""

import argparse
import hashlib
import io
import json
import os
import random
import re
import time
from collections import Counter, defaultdict

import torch
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PARSER_VERSION = "v1"

T0 = time.time()


def log(msg):
    """Timestamped, immediately flushed log line (so `tail -f` works under Slurm)."""
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)

BOX_COLS = ["box_xmin", "box_ymin", "box_xmax", "box_ymax"]  # normalised [0,1] in RefOI-TLHF

PROMPT_DESC = (
    "Identify the target in the provided image given the description. "
    "Output your answer in a bounding box in the format [Xmin, Ymin, Xmax, Ymax]. "
    'Description: "{desc}"'
)
PROMPT_NONE = (
    "Identify the target in the provided image. "
    "Output your answer in a bounding box in the format [Xmin, Ymin, Xmax, Ymax]."
)

# ----------------------------------------------------------------------------- red-box cleaning
# RefOI-TLHF descriptions were generated on images with a red box drawn around the target
# ("The object in the red box is a ...", "The red-boxed object is ..."). The listener sees
# the clean image, so these references could be replaced by "the target". This is only applied
# with --clean-red-box; by default the descriptions are passed to the model verbatim.
_NOUN = r"(?:object|item|thing|target|animal|person|subject|area|region)"
_BOXWORD = r"(?:red|highlighted)[\s-]*(?:bounding\s+)?(?:box|rectangle|square|outline|frame|border)"
RE_IN_BOX = re.compile(
    rf"\b(?:the\s+)?{_NOUN}\s+(?:inside|in|within|enclosed\s+(?:in|by)|marked\s+by|highlighted\s+by)"
    rf"\s+the\s+{_BOXWORD}",
    re.I,
)
RE_BOXED = re.compile(
    rf"\bthe\s+(?:red[\s-]*boxed|red[\s-]*outlined|red[\s-]*highlighted)\s+{_NOUN}", re.I
)
RE_LEFTOVER = re.compile(
    r"\bred[\s-]*(?:bounding\s+)?(?:box|boxed|rectangle|square|outline|outlined|frame|border)", re.I
)


def _target(m):
    return "The target" if m.group(0)[0].isupper() else "the target"


def clean_description(text, keep_red_box=False):
    t = text.strip()
    if keep_red_box:
        return t
    t = RE_IN_BOX.sub(_target, t)
    t = RE_BOXED.sub(_target, t)
    return t.strip()


def has_red_leak(text):
    """True if the (cleaned) description still refers to a red box."""
    return bool(RE_LEFTOVER.search(text))


# ----------------------------------------------------------------------------- pass 1
def load_meta(dataset, split, desc_col, cat_col, extra_cols):
    cols = [desc_col, cat_col] + BOX_COLS + extra_cols
    ds = (
        load_dataset(dataset, split=split, streaming=True)
        .select_columns(["image"] + cols)
        .cast_column("image", HFImage(decode=False))
    )
    rows = []
    for i, ex in enumerate(ds):
        ex = dict(ex)
        ex["idx"] = i
        # RefOI-TLHF has no image identifier: hash the image bytes (same image -> same id)
        ex["image_id"] = hashlib.sha1(ex.pop("image")["bytes"]).hexdigest()[:16]
        # "key" identifies the target (category + gold box), used to pick shuffled descriptions
        ex["key"] = f'{ex[cat_col]}|' + "|".join(f"{ex[c]:.4f}" for c in BOX_COLS)
        rows.append(ex)
    print(f"[pass 1] {len(rows)} rows, {len({r['key'] for r in rows})} distinct targets")
    return rows


def build_shuffles(rows, cat_col, style_col, rng):
    """For each row pick a description from a different target (same style when possible)."""
    by_cat_style = defaultdict(list)
    by_style = defaultdict(list)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat_style[(r[cat_col], r.get(style_col))].append(r)
        by_style[r.get(style_col)].append(r)
        by_cat[r[cat_col]].append(r)

    def pick(pool, key):
        cands = [c for c in pool if c["key"] != key]
        return rng.choice(cands) if cands else None

    out = {}
    for r in rows:
        same = pick(by_cat_style[(r[cat_col], r.get(style_col))], r["key"])
        if same is None:  # fall back to same category, any style
            same = pick(by_cat[r[cat_col]], r["key"])
        other = pick(by_style[r.get(style_col)], r["key"])
        out[r["idx"]] = {"shuffled_same_cat": same, "shuffled_random": other}
    return out


# ----------------------------------------------------------------------------- model
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


_N = r"(-?\d+(?:\.\d+)?)"
BOX_RE = re.compile(rf"\[\s*{_N}\s*,\s*{_N}\s*,\s*{_N}\s*,\s*{_N}\s*\]")


def parse_box(text, w, h):
    """Read the first [x1, y1, x2, y2] list in the answer.

    Qwen2.5-VL answers in absolute pixels of the (resized) input image -> normalise by (w, h).
    Returns (box or None, status) with status in {"ok", "no_box", "invalid_box"}.
    The text is usually truncated JSON, so json.loads is deliberately not used.
    """
    m = BOX_RE.search(text)
    if not m:
        return None, "no_box"
    x1, y1, x2, y2 = (float(v) for v in m.groups())
    if max(x1, y1, x2, y2) <= 1.0:  # already normalised
        box = [x1, y1, x2, y2]
    else:
        box = [x1 / w, y1 / h, x2 / w, y2 / h]
    box = [min(1.0, max(0.0, v)) for v in box]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None, "invalid_box"
    return box, "ok"


class Listener:
    def __init__(self, model_name, min_pixels, max_pixels, max_new_tokens=64):
        self.max_new_tokens = max_new_tokens
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
        ).eval()
        self.proc = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

    @torch.no_grad()
    def __call__(self, image, prompt):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        _, gh, gw = inputs["image_grid_thw"][0].tolist()
        rh, rw = gh * 14, gw * 14  # size of the image actually seen by the model
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        box, status = parse_box(gen, rw, rh)
        return gen, box, status, rw, rh


# ----------------------------------------------------------------------------- summary
def summarize(path):
    stats = defaultdict(lambda: {"n": 0, "ok": 0, "iou": 0.0, "s50": 0, "s75": 0, "iou_ok": 0.0, "s50_ok": 0})
    ignored = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("parser") != PARSER_VERSION:
                ignored += 1
                continue
            s = stats[r["condition"]]
            s["n"] += 1
            s["iou"] += r["iou"]
            s["s50"] += r["success50"]
            s["s75"] += r["success75"]
            if r["parsed"]:
                s["ok"] += 1
                s["iou_ok"] += r["iou"]
                s["s50_ok"] += r["success50"]
    if ignored:
        print(f"[summary] ignored {ignored} records from an older parser")
    print(f"{'condition':<20}{'n':>6}{'box%':>8}{'mIoU':>8}{'acc50':>8}{'acc75':>8}{'mIoU|box':>10}{'acc50|box':>11}")
    for cond, s in stats.items():
        n, ok = s["n"], s["ok"]
        print(
            f"{cond:<20}{n:>6}{100 * ok / n:>7.1f}%{s['iou'] / n:>8.3f}{s['s50'] / n:>8.3f}{s['s75'] / n:>8.3f}"
            f"{(s['iou_ok'] / ok if ok else 0):>10.3f}{(s['s50_ok'] / ok if ok else 0):>11.3f}"
        )


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Seed42Lab/RefOI-TLHF")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--desc-col", default="full_description")
    ap.add_argument("--cat-col", default="label_name")
    ap.add_argument("--style-col", default="note", help="default_prompt / brief_prompt")
    ap.add_argument("--distractor-col", default="co_occurrence")
    ap.add_argument("--source-col", default="source")
    ap.add_argument("--n-items", type=int, default=2000, help="random sample of rows (0 = all)")
    ap.add_argument("--distractors-gt", type=float, default=None,
                    help="only sample rows whose distractor column (co_occurrence) is > this value, e.g. 1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--with-category-only", action="store_true")
    ap.add_argument("--clean-red-box", action="store_true",
                    help="rewrite 'the object in the red box' as 'the target' (default: descriptions are used verbatim)")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--out", default="results.jsonl")
    args = ap.parse_args()

    if args.summarize_only:
        summarize(args.out)
        return

    rng = random.Random(args.seed)
    extra = [args.style_col, args.distractor_col, args.source_col]
    rows = load_meta(args.dataset, args.split, args.desc_col, args.cat_col, extra)
    shuffles = build_shuffles(rows, args.cat_col, args.style_col, rng)
    log("pass 1 + shuffles done")
    dist = Counter(r.get(args.distractor_col) for r in rows)
    print(f"[pass 1] {args.distractor_col} distribution:",
          dict(sorted(dist.items(), key=lambda kv: float(kv[0]) if isinstance(kv[0], (int, float)) else float("inf"))),
          flush=True)

    # The dataset is sorted by speaker model, so sample at random over the whole split
    # (shuffles are built from all rows first, so the filter does not change the shuffled descriptions)
    def n_distractors(r):
        try:
            return float(r.get(args.distractor_col))
        except (TypeError, ValueError):
            return None

    if args.distractors_gt is None:
        all_idx = list(range(len(rows)))
    else:
        all_idx = [r["idx"] for r in rows
                   if n_distractors(r) is not None and n_distractors(r) > args.distractors_gt]
        print(f"[filter] {args.distractor_col} > {args.distractors_gt:g}: {len(all_idx)} of {len(rows)} rows",
              flush=True)
    chosen = set(rng.sample(all_idx, args.n_items) if 0 < args.n_items < len(all_idx) else all_idx)
    print(f"[sample] {len(chosen)} items selected", flush=True)

    done, ignored = set(), 0
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                r = json.loads(line)
                if r.get("parser") != PARSER_VERSION:
                    ignored += 1
                    continue
                done.add((r["idx"], r["condition"]))
    print(f"[resume] {len(done)} (item, condition) pairs already done")
    if ignored:
        print(f"[resume] WARNING: {ignored} records from an older parser ignored; use a new --out file")

    listener = Listener(args.model, args.min_pixels, args.max_pixels, args.max_new_tokens)
    log("model loaded")

    # ---- pass 2: stream images, decode only the items that still have work to do
    ds = (
        load_dataset(args.dataset, split=args.split, streaming=True)
        .select_columns(["image", args.cat_col])
        .cast_column("image", HFImage(decode=False))
    )
    remaining = set(chosen)
    fout = open(args.out, "a")
    t0, n_done_items = time.time(), 0

    for i, ex in enumerate(ds):
        if i % 500 == 0:
            log(f"pass 2: at row {i}")
        if i not in chosen:
            continue
        meta = rows[i]
        assert ex[args.cat_col] == meta[args.cat_col], "pass-1 / pass-2 order mismatch"
        gold = [meta["box_xmin"], meta["box_ymin"], meta["box_xmax"], meta["box_ymax"]]

        conds = {"real": meta, "none": None}
        conds["shuffled_same_cat"] = shuffles[i]["shuffled_same_cat"]
        conds["shuffled_random"] = shuffles[i]["shuffled_random"]
        if args.with_category_only:
            conds["category_only"] = {"_literal": f"a {meta[args.cat_col]}", "idx": -1}

        todo = [c for c, src in conds.items() if (i, c) not in done and (c == "none" or src is not None)]
        if todo:
            image = Image.open(io.BytesIO(ex["image"]["bytes"])).convert("RGB")
            for cond in todo:
                src = conds[cond]
                if cond == "none":
                    prompt, desc, src_idx = PROMPT_NONE, "", -1
                else:
                    if "_literal" in src:
                        desc = src["_literal"]
                    else:
                        desc = clean_description(src[args.desc_col], keep_red_box=not args.clean_red_box)
                    prompt, src_idx = PROMPT_DESC.format(desc=desc), src["idx"]

                tg = time.time()
                try:
                    gen, box, status, rw, rh = listener(image, prompt)
                except Exception as e:  # e.g. CUDA OOM on a huge image: skip, do not record
                    print(f"  [skip] item {i} cond {cond}: {type(e).__name__}: {e}")
                    torch.cuda.empty_cache()
                    continue

                log(f"item {i} {cond}: {time.time() - tg:.1f}s, status={status}")
                s = iou(box, gold) if box else 0.0
                fout.write(json.dumps({
                    "parser": PARSER_VERSION,
                    "idx": i, "image_id": meta["image_id"], "key": meta["key"], "condition": cond,
                    "speaker": meta.get(args.source_col), "style": meta.get(args.style_col),
                    "n_distractors": meta.get(args.distractor_col),
                    "category": meta[args.cat_col],
                    "description": desc, "desc_source_idx": src_idx,
                    "red_leak": has_red_leak(desc),
                    "raw_output": gen, "status": status, "parsed": box is not None,
                    "pred": box, "gold": gold, "rw": rw, "rh": rh,
                    "iou": s, "success50": int(s >= 0.5), "success75": int(s >= 0.75),
                }) + "\n")
            fout.flush()
            n_done_items += 1
            if n_done_items % 25 == 0:
                el = time.time() - t0
                print(f"  {n_done_items} items in {el / 60:.1f} min ({el / n_done_items:.1f} s/item), "
                      f"{len(remaining) - 1} left")

        remaining.discard(i)
        if not remaining:
            break

    fout.close()
    print("done ->", args.out)
    summarize(args.out)


if __name__ == "__main__":
    main()
    
