"""T5 - Reproducible error analysis of ASR prediction files on the frozen test split.

Reads results/predictions/<RUN>.tsv (id, reference, hypothesis), checks them against
data/splits/test.tsv exactly like score_predictions.py, then writes to --output-dir:

  summary.csv        per run: pooled WER/CER, S/D/I, insertion share, median utterance WER,
                     utterances with WER > 1, repetition loops, (near-)empty outputs, perfect outputs
  utterances.tsv     per run x utterance: counts, WER, CER, loop length and test-set metadata
  subgroups.csv      pooled WER/CER by gender, clip duration, digits in reference, loudness (RMS tercile)
  pairwise.csv       per-utterance WER comparison of each run against the first run given
  substitutions.tsv  most frequent (reference word -> hypothesis word) substitutions per run
  examples.md        best / median / worst utterances per run for the qualitative audit
  bootstrap.csv      95% bootstrap CI of WER per run and paired WER differences between runs; utterances
                     are resampled by sentence_id because FLEURS has several speakers per sentence
  oov.csv            reference-word error rate for words seen vs never seen in train transcripts
  word_boundary.csv  CER with/without spaces and the share of character edits that are space edits
                     (merged/split words, frequent in agglutinative Swahili)
  substitution_categories.csv  substituted word pairs by where they differ: short function word
                     (na/ya/za...), word start (noun-class/subject prefix), word end (suffix), middle,
                     or other (unrelated word)

All numbers use metrics.normalize_transcript, the same normalisation as the benchmark table.
Usage:
  python src/error_analysis.py --predictions results/predictions/E1.tsv results/predictions/E2.tsv \
      results/predictions/E3.tsv
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path

import jiwer

from metrics import evaluate_asr, normalize_transcript
from run_baseline import ID_COL
from score_predictions import check_alignment, read_tsv

LOOP_MIN_REPEATS = 4   # an n-gram (n<=3) repeated back-to-back this often counts as a hallucination loop
LOOP_MAX_N = 3


# ------------------------------------------------------------------ per utterance
def utterance_errors(reference: str, hypothesis: str) -> dict:
    ref, hyp = normalize_transcript(reference), normalize_transcript(hypothesis)
    out = jiwer.process_words(ref, hyp)
    ref_words, hyp_words = out.references[0], out.hypotheses[0]
    pairs, ref_ops = [], []
    for chunk in out.alignments[0]:
        if chunk.type != "insert":
            ref_ops += [(w, chunk.type) for w in ref_words[chunk.ref_start_idx:chunk.ref_end_idx]]
        if chunk.type == "substitute":
            pairs += list(zip(ref_words[chunk.ref_start_idx:chunk.ref_end_idx],
                              hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]))
    n_ref = len(ref_words)
    return {
        "n_ref": n_ref,
        "hyp_words": len(hyp_words),
        "sub": out.substitutions,
        "del": out.deletions,
        "ins": out.insertions,
        "wer": (out.substitutions + out.deletions + out.insertions) / n_ref,
        "cer": jiwer.cer(ref, hyp),
        "loop": longest_repeat_run(hyp_words),
        "pairs": pairs,
        "ref_ops": ref_ops,
    }


def longest_repeat_run(words: list[str], max_n: int = LOOP_MAX_N) -> int:
    """Longest number of back-to-back repeats of any 1..max_n-gram ("na na na" -> 3)."""
    if not words:
        return 0
    best = 1
    for n in range(1, max_n + 1):
        for start in range(len(words) - n + 1):
            gram, run, pos = words[start:start + n], 1, start + n
            while words[pos:pos + n] == gram:
                run, pos = run + 1, pos + n
            best = max(best, run)
    return best


# ------------------------------------------------------------------ grouping
def duration_bucket(seconds: float) -> str:
    if seconds < 10:
        return "<10s"
    if seconds < 15:
        return "10-15s"
    if seconds < 20:
        return "15-20s"
    return ">=20s"


def tercile_labels(values: list[float]) -> list[str]:
    ordered = sorted(values)
    lo, hi = ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3]
    return ["low" if v < lo else "mid" if v < hi else "high" for v in values]


def pooled_wer(rows: list[dict]) -> float:
    return sum(r["sub"] + r["del"] + r["ins"] for r in rows) / sum(r["n_ref"] for r in rows)


def top_substitutions(rows: list[dict], k: int) -> list[tuple[tuple[str, str], int]]:
    return Counter(p for r in rows for p in r["pairs"]).most_common(k)


# ------------------------------------------------------------------ bootstrap
def _cluster_totals(rows: list[dict]) -> tuple[list[str], list[tuple[int, int]]]:
    totals: dict[str, list[int]] = {}
    for r in rows:
        t = totals.setdefault(r["sentence_id"], [0, 0])
        t[0] += r["sub"] + r["del"] + r["ins"]
        t[1] += r["n_ref"]
    return list(totals), [tuple(v) for v in totals.values()]


def _percentile_ci(stats: list[float]) -> tuple[float, float]:
    stats = sorted(stats)
    n = len(stats) - 1
    return stats[math.floor(0.025 * n)], stats[math.ceil(0.975 * n)]


def bootstrap_ci(rows: list[dict], n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """95% percentile CI of pooled WER, resampling whole sentence_id clusters with replacement."""
    _, totals = _cluster_totals(rows)
    rng, k, stats = random.Random(seed), len(totals), []
    for _ in range(n_boot):
        draw = [totals[rng.randrange(k)] for _ in range(k)]
        stats.append(sum(e for e, _ in draw) / sum(n for _, n in draw))
    return _percentile_ci(stats)


def paired_bootstrap(rows_a: list[dict], rows_b: list[dict], n_boot: int = 2000, seed: int = 42) -> dict:
    """WER(b) - WER(a) with a paired, sentence-clustered bootstrap CI and two-sided p-value."""
    if [r["sentence_id"] for r in rows_a] != [r["sentence_id"] for r in rows_b]:
        raise ValueError("paired bootstrap needs both runs in the same utterance order")
    _, ta = _cluster_totals(rows_a)
    _, tb = _cluster_totals(rows_b)
    rng, k, diffs = random.Random(seed), len(ta), []
    for _ in range(n_boot):
        idx = [rng.randrange(k) for _ in range(k)]
        n = sum(ta[i][1] for i in idx)
        diffs.append((sum(tb[i][0] for i in idx) - sum(ta[i][0] for i in idx)) / n)
    delta = pooled_wer(rows_b) - pooled_wer(rows_a)
    low, high = _percentile_ci(diffs)
    le, ge = sum(d <= 0 for d in diffs), sum(d >= 0 for d in diffs)
    p_value = min(1.0, 2 * (min(le, ge) + 1) / (n_boot + 1))
    return {"delta": delta, "ci_low": low, "ci_high": high, "p_value": p_value}


# ------------------------------------------------------------------ vocabulary, boundaries, morphology
def train_vocabulary(train_manifest: Path) -> set[str]:
    return {w for r in read_tsv(train_manifest) for w in normalize_transcript(r["raw_transcription"]).split()}


def oov_error_rates(rows: list[dict], train_vocab: set[str]) -> dict[str, dict[str, int]]:
    """Reference-word errors (substituted or deleted) split by whether the word occurs in train."""
    out = {"in_vocab": {"ref_words": 0, "errors": 0}, "oov": {"ref_words": 0, "errors": 0}}
    for r in rows:
        for word, op in r["ref_ops"]:
            bucket = out["in_vocab" if word in train_vocab else "oov"]
            bucket["ref_words"] += 1
            bucket["errors"] += op != "equal"
    return out


def space_edits(reference: str, hypothesis: str) -> tuple[int, int]:
    """(character edits involving a space, all character edits) on normalised text."""
    ref, hyp = normalize_transcript(reference), normalize_transcript(hypothesis)
    out = jiwer.process_characters(ref, hyp)
    r_chars, h_chars, space = out.references[0], out.hypotheses[0], 0
    for c in out.alignments[0]:
        r_span, h_span = r_chars[c.ref_start_idx:c.ref_end_idx], h_chars[c.hyp_start_idx:c.hyp_end_idx]
        if c.type == "delete":
            space += r_span.count(" ")
        elif c.type == "insert":
            space += h_span.count(" ")
        elif c.type == "substitute":
            space += sum(a == " " or b == " " for a, b in zip(r_span, h_span))
    return space, out.substitutions + out.deletions + out.insertions


def classify_substitution(ref_word: str, hyp_word: str) -> str:
    """Where two substituted words differ. Heuristic, not a morphological parser."""
    if len(ref_word) <= 3 and len(hyp_word) <= 3:
        return "function_word"
    prefix = 0
    while prefix < min(len(ref_word), len(hyp_word)) and ref_word[prefix] == hyp_word[prefix]:
        prefix += 1
    suffix, limit = 0, min(len(ref_word), len(hyp_word)) - prefix
    while suffix < limit and ref_word[-1 - suffix] == hyp_word[-1 - suffix]:
        suffix += 1
    if prefix < 2 and suffix < 2:
        return "other"   # nothing meaningful shared at either edge: a different word
    if prefix == 0:
        return "start"
    if suffix == 0:
        return "end"
    return "middle"


# ------------------------------------------------------------------ report
def _write_csv(path: Path, rows: list[dict], delimiter: str = ",", fields: list[str] | None = None) -> None:
    """Write rows; `fields` gives the header when a table can legitimately be empty."""
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else fields, delimiter=delimiter, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _example_block(run: str, rows: list[dict], k: int = 3) -> list[str]:
    ordered = sorted(rows, key=lambda r: (r["wer"], r["id"]))
    mid = len(ordered) // 2
    picks = [("Best", ordered[:k]), ("Median", ordered[mid - k // 2: mid - k // 2 + k]), ("Worst", ordered[-k:][::-1])]
    lines = [f"## {run}", ""]
    for label, chosen in picks:
        lines.append(f"### {label}")
        for r in chosen:
            hyp = r["hypothesis"] if len(r["hypothesis"]) <= 300 else r["hypothesis"][:300] + " …"
            lines += [f"- `{r['id']}` WER {r['wer']:.2f}", f"  - REF: {r['reference']}", f"  - HYP: {hyp}"]
        lines.append("")
    return lines


def analyse(prediction_paths: list[Path], test_manifest: Path, output_dir: Path,
            train_manifest: Path | None = None, n_boot: int = 2000, seed: int = 42) -> None:
    test_rows = read_tsv(test_manifest)
    meta = {r[ID_COL]: r for r in test_rows}
    order = [r[ID_COL] for r in test_rows]
    loudness = dict(zip(order, tercile_labels([float(meta[i]["rms"]) for i in order])))

    per_run: dict[str, list[dict]] = {}
    for path in prediction_paths:
        run = path.stem
        preds = read_tsv(path)
        check_alignment(test_rows, preds, run)
        by_id = {r["id"]: r for r in preds}
        rows = []
        for uid in order:
            p, m = by_id[uid], meta[uid]
            rows.append({
                "run": run, "id": uid, "sentence_id": m["sentence_id"],
                "reference": p["reference"], "hypothesis": p["hypothesis"],
                "gender": m["gender"],
                "duration": duration_bucket(float(m["duration_seconds"])),
                "digits": "yes" if re.search(r"\d", p["reference"]) else "no",
                "loudness": loudness[uid],
                **utterance_errors(p["reference"], p["hypothesis"]),
            })
        per_run[run] = rows

    output_dir.mkdir(parents=True, exist_ok=True)
    summary, subgroups, subs, pairwise, utts, examples = [], [], [], [], [], ["# Error-analysis examples", ""]
    for run, rows in per_run.items():
        s, d, i = (sum(r[k] for r in rows) for k in ("sub", "del", "ins"))
        official = evaluate_asr([r["reference"] for r in rows], [r["hypothesis"] for r in rows])
        summary.append({
            "run": run, "n_utts": len(rows),
            "wer": round(pooled_wer(rows), 4), "cer": round(official["cer"], 4),
            "sub": s, "del": d, "ins": i, "ins_share": round(i / (s + d + i), 4) if s + d + i else 0.0,
            "median_utt_wer": round(statistics.median(r["wer"] for r in rows), 4),
            "utts_wer_gt1": sum(r["wer"] > 1 for r in rows),
            "loop_utts": sum(r["loop"] >= LOOP_MIN_REPEATS for r in rows),
            "le1_word_utts": sum(r["hyp_words"] <= 1 for r in rows),
            "perfect_utts": sum(r["wer"] == 0 for r in rows),
            "wer_excl_wer_gt1": round(pooled_wer(kept), 4) if (kept := [r for r in rows if r["wer"] <= 1]) else "",
        })
        for factor in ("gender", "duration", "digits", "loudness"):
            for level in sorted({r[factor] for r in rows}):
                grp = [r for r in rows if r[factor] == level]
                subgroups.append({
                    "run": run, "factor": factor, "level": level, "n_utts": len(grp),
                    "ref_words": sum(r["n_ref"] for r in grp), "wer": round(pooled_wer(grp), 4),
                    "cer": round(evaluate_asr([r["reference"] for r in grp], [r["hypothesis"] for r in grp])["cer"], 4),
                    "loop_utts": sum(r["loop"] >= LOOP_MIN_REPEATS for r in grp),
                })
        for (ref_w, hyp_w), n in top_substitutions(rows, 25):
            subs.append({"run": run, "reference_word": ref_w, "hypothesis_word": hyp_w, "count": n})
        utts += [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("pairs", "ref_ops")}
                 for r in rows]
        examples += _example_block(run, rows)

    runs = list(per_run)
    for other in runs[1:]:
        a, b = per_run[runs[0]], per_run[other]
        pairwise.append({
            "baseline_run": runs[0], "run": other,
            "run_better": sum(y["wer"] < x["wer"] for x, y in zip(a, b)),
            "run_worse": sum(y["wer"] > x["wer"] for x, y in zip(a, b)),
            "equal": sum(y["wer"] == x["wer"] for x, y in zip(a, b)),
        })

    # ---- extended analyses
    boot = []
    for run, rows in per_run.items():
        low, high = bootstrap_ci(rows, n_boot, seed)
        boot.append({"kind": "run", "baseline_run": "", "run": run, "wer": round(pooled_wer(rows), 4),
                     "ci_low": round(low, 4), "ci_high": round(high, 4), "p_value": ""})
    for i, a in enumerate(runs):
        for b in runs[i + 1:]:
            res = paired_bootstrap(per_run[a], per_run[b], n_boot, seed)
            boot.append({"kind": "pair", "baseline_run": a, "run": b, "wer": round(res["delta"], 4),
                         "ci_low": round(res["ci_low"], 4), "ci_high": round(res["ci_high"], 4),
                         "p_value": round(res["p_value"], 4)})
    _write_csv(output_dir / "bootstrap.csv", boot)

    boundary, categories = [], []
    for run, rows in per_run.items():
        edits = [space_edits(r["reference"], r["hypothesis"]) for r in rows]
        sp, tot = sum(e[0] for e in edits), sum(e[1] for e in edits)
        refs = [normalize_transcript(r["reference"]).replace(" ", "") for r in rows]
        hyps = [normalize_transcript(r["hypothesis"]).replace(" ", "") for r in rows]
        boundary.append({"run": run, "cer": round(evaluate_asr([r["reference"] for r in rows],
                                                               [r["hypothesis"] for r in rows])["cer"], 4),
                         "cer_no_space": round(jiwer.cer(refs, hyps), 4), "space_edits": sp, "char_edits": tot,
                         "space_edit_share": round(sp / tot, 4) if tot else 0.0})
        cat = Counter(classify_substitution(a, b) for r in rows for a, b in r["pairs"])
        total = sum(cat.values())
        for name in ("function_word", "start", "middle", "end", "other"):
            categories.append({"run": run, "category": name, "count": cat[name],
                               "share": round(cat[name] / total, 4) if total else 0.0})
    _write_csv(output_dir / "word_boundary.csv", boundary)
    _write_csv(output_dir / "substitution_categories.csv", categories)

    if train_manifest is not None:
        vocab, oov = train_vocabulary(train_manifest), []
        for run, rows in per_run.items():
            for status, t in oov_error_rates(rows, vocab).items():
                oov.append({"run": run, "vocab_status": status, "ref_words": t["ref_words"], "errors": t["errors"],
                            "error_rate": round(t["errors"] / t["ref_words"], 4) if t["ref_words"] else 0.0})
        _write_csv(output_dir / "oov.csv", oov)

    _write_csv(output_dir / "summary.csv", summary)
    _write_csv(output_dir / "subgroups.csv", subgroups)
    _write_csv(output_dir / "substitutions.tsv", subs, "\t",
               fields=["run", "reference_word", "hypothesis_word", "count"])
    _write_csv(output_dir / "utterances.tsv", utts, "\t")
    if pairwise:
        _write_csv(output_dir / "pairwise.csv", pairwise)
    (output_dir / "examples.md").write_text("\n".join(examples), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-manifest", type=Path, default=Path("data/splits/test.tsv"))
    p.add_argument("--predictions", type=Path, nargs="+", required=True,
                   help="prediction TSVs; the first one is the reference run for pairwise.csv")
    p.add_argument("--output-dir", type=Path, default=Path("results/error_analysis"))
    p.add_argument("--train-manifest", type=Path, default=Path("data/splits/train.tsv"),
                   help="train transcripts used only to build the vocabulary for oov.csv")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    analyse(args.predictions, args.test_manifest, args.output_dir, args.train_manifest, args.n_boot, args.seed)
    with (args.output_dir / "summary.csv").open(encoding="utf-8") as f:
        print(f.read().rstrip())
    print(f"Wrote error analysis to {args.output_dir}/")


if __name__ == "__main__":
    main()
