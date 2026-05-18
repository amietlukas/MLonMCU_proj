"""
Aggregate per-model results CSVs into one comparison table.

For each row, from results_<tag>.csv we compute:
  n_samples, accuracy, macro_precision, macro_recall, macro_f1,
  per-phase timings (pre / infer / post) mean + p50/p95 for infer,
  energy per inference (mJ) assuming the STM32U585 Run1@160 MHz typical IDD.

Static model info (params, MACC, flash, RAM) comes from `stedgeai analyze`
on each ONNX (captured in STATIC_INFO below).

Run:
    python firmware/Host/compare_models.py
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

REPO_ROOT = Path("/mnt/core/MLonMCU_proj")
OUT_CSV   = REPO_ROOT / "firmware" / "Host" / "model_comparison.csv"

CLASS_NAMES = ["palm", "rock", "pinkie", "one", "fist", "others"]

# (display tag, run dir relative, results-csv filename, "currently in flash?")
ROWS = [
    ("smallnet_fp32",
     "software/classifier/runs/smallnet_greyscale-20260504-184641_BASELINE_SMALL",
     "results_smallnet_fp32.csv",       False),
    ("smallnet_int8",
     "software/classifier/runs/smallnet_greyscale-20260504-184641_BASELINE_SMALL",
     "results_smallnet_int8.csv",       True),
    ("bignet_fp32",
     "software/classifier/runs/bignet-20260505-163101_BASELINE",
     "results_bignet_fp32.csv",         False),
    ("bignet_int8",
     "software/classifier/runs/bignet-20260505-163101_BASELINE",
     "results_bignet_int8.csv",         False),
    ("bignet_pruned_fp32",
     "software/classifier/runs/bignet_pruned-20260505-225640_BASELINE_PRUNED",
     "results_bignet_pruned_fp32.csv",  False),
    ("bignet_pruned_int8",
     "software/classifier/runs/bignet_pruned-20260505-225640_BASELINE_PRUNED",
     "results_bignet_pruned_int8.csv",  False),
]

LEGACY_ALIAS = {
    "bignet_pruned_int8": "results_bignetpruned.csv",
}

# `stedgeai analyze --target stm32u5` values per model.
# weights_B == flash footprint of the weights (ROM).
# acts_B    == X-CUBE-AI `ram (total)` = peak activation buffer in RAM
#              (with --allocate-inputs --allocate-outputs, IO is folded in).
STATIC_INFO = {
    "smallnet_fp32":      {"params":  93446, "macc": 184_666_598, "weights_B": 373_784, "acts_B": 707_108},
    "smallnet_int8":      {"params":  93446, "macc": 183_668_710, "weights_B":  95_160, "acts_B": 184_100},
    "bignet_fp32":        {"params": 190182, "macc":  72_482_390, "weights_B": 760_728, "acts_B": 353_252},
    "bignet_int8":        {"params": 190182, "macc":  71_911_382, "weights_B": 192_232, "acts_B":  93_888},
    "bignet_pruned_fp32": {"params":  17417, "macc":   7_614_225, "weights_B":  69_668, "acts_B": 109_976},
    "bignet_pruned_int8": {"params":  17417, "macc":   7_437_357, "weights_B":  18_052, "acts_B":  27_640},
}

# STM32U585 datasheet DS13086 (Rev 10), Run1 mode, fHCLK = 160 MHz,
# VCORE = 1.2 V (Range 1), VDD = 3.3 V:
#   typical dynamic IDD = 31.9 µA/MHz  (datasheet headline figure)
# => IDD     = 31.9 µA/MHz x 160 MHz = 5.104 mA
# => P_typ   = 3.3 V x 5.104 mA      = 16.843 mW
# Energy per inference = P_typ x (pre_ms + infer_ms + post_ms) / 1000   [mJ]
MCU_RUN_IDD_UA_PER_MHZ = 31.9
MCU_CLK_MHZ            = 160.0
MCU_VDD_V              = 3.3
MCU_POWER_MW           = MCU_VDD_V * (MCU_RUN_IDD_UA_PER_MHZ * MCU_CLK_MHZ) / 1000.0  # ~16.843 mW


# ---------- helpers ----------
def percentile(xs, p):
    if not xs:
        return float("nan")
    xs2 = sorted(xs)
    k = (len(xs2) - 1) * p / 100.0
    f = int(k); c = min(f + 1, len(xs2) - 1)
    return xs2[f] + (xs2[c] - xs2[f]) * (k - f)


def load_results(path: Path):
    if not path.exists():
        return None
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def confusion(rows, names):
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    cm = [[0] * n for _ in range(n)]
    for r in rows:
        g = idx.get(r["gt"]); p = idx.get(r["pred"])
        if g is None or p is None:
            continue
        cm[g][p] += 1
    return cm


def macro_prf1(cm):
    """Return (macro_precision, macro_recall, macro_f1)."""
    n = len(cm)
    precs, recs, f1s = [], [], []
    for i in range(n):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i][c] for c in range(n)) - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precs.append(p); recs.append(r); f1s.append(f)
    return sum(precs) / n, sum(recs) / n, sum(f1s) / n


def _float_col(rows, key):
    return [float(r[key]) for r in rows if r.get(key, "").strip()]


def aggregate(rows):
    n = len(rows)
    correct = sum(int(r["correct"]) for r in rows)
    # Time fields are blank for host-side ONNX eval (software/classifier/test_onnx.py).
    pre = _float_col(rows, "pre_ms")
    inf = _float_col(rows, "infer_ms")
    pos = _float_col(rows, "post_ms")
    tot = _float_col(rows, "total_ms")
    cm  = confusion(rows, CLASS_NAMES)
    mp, mr, mf1 = macro_prf1(cm)

    # Energy per inference: average pre+infer+post if all three are available, else NaN.
    if pre and inf and pos and len(pre) == len(inf) == len(pos):
        total_per_sample = [p + i + q for p, i, q in zip(pre, inf, pos)]
        energy_mJ = MCU_POWER_MW * statistics.fmean(total_per_sample) / 1000.0
    else:
        energy_mJ = float("nan")

    return {
        "n":              n,
        "acc":            correct / n if n else float("nan"),
        "macro_prec":     mp,
        "macro_rec":      mr,
        "macro_f1":       mf1,
        "pre_ms_mean":    statistics.fmean(pre) if pre else float("nan"),
        "infer_mean":     statistics.fmean(inf) if inf else float("nan"),
        "post_ms_mean":   statistics.fmean(pos) if pos else float("nan"),
        "infer_p50":      percentile(inf, 50)   if inf else float("nan"),
        "infer_p95":      percentile(inf, 95)   if inf else float("nan"),
        "total_mean":     statistics.fmean(tot) if tot else float("nan"),
        "energy_mJ":      energy_mJ,
    }


def fmt(v, spec=""):
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:           # NaN
            return ""
        return format(v, spec)
    return str(v)


FIELDS = [
    "model", "n_samples",
    "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "pre_ms_mean", "infer_ms_mean", "post_ms_mean",
    "infer_ms_p50", "infer_ms_p95",
    "params", "macc", "flash_KB", "ram_KB",
    "energy_mJ",
    "results_csv",
]


# ---------- build table ----------
def main():
    table = []
    for tag, run_rel, fname, _flashed in ROWS:
        metrics_dir = REPO_ROOT / run_rel / "metrics"
        res_path = metrics_dir / fname
        if not res_path.exists() and tag in LEGACY_ALIAS:
            res_path = metrics_dir / LEGACY_ALIAS[tag]

        rows = load_results(res_path)
        agg = aggregate(rows) if rows else None
        st  = STATIC_INFO.get(tag, {})

        table.append({
            "model":           tag,
            "n_samples":       agg["n"] if agg else 0,
            "accuracy":        agg["acc"] if agg else None,
            "macro_precision": agg["macro_prec"] if agg else None,
            "macro_recall":    agg["macro_rec"]  if agg else None,
            "macro_f1":        agg["macro_f1"]   if agg else None,
            "pre_ms_mean":     agg["pre_ms_mean"]  if agg else None,
            "infer_ms_mean":   agg["infer_mean"]   if agg else None,
            "post_ms_mean":    agg["post_ms_mean"] if agg else None,
            "infer_ms_p50":    agg["infer_p50"]    if agg else None,
            "infer_ms_p95":    agg["infer_p95"]    if agg else None,
            "params":          st.get("params"),
            "macc":            st.get("macc"),
            "flash_KB":        (st["weights_B"] / 1024) if st.get("weights_B") else None,
            "ram_KB":          (st["acts_B"]    / 1024) if st.get("acts_B")    else None,
            "energy_mJ":       agg["energy_mJ"] if agg else None,
            "results_csv":     str(res_path) if rows else "(missing)",
        })

    # --- save CSV ---
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in table:
            w.writerow({
                "model":           row["model"],
                "n_samples":       row["n_samples"],
                "accuracy":        fmt(row["accuracy"], ".4f"),
                "macro_precision": fmt(row["macro_precision"], ".4f"),
                "macro_recall":    fmt(row["macro_recall"], ".4f"),
                "macro_f1":        fmt(row["macro_f1"], ".4f"),
                "pre_ms_mean":     fmt(row["pre_ms_mean"], ".3f"),
                "infer_ms_mean":   fmt(row["infer_ms_mean"], ".2f"),
                "post_ms_mean":    fmt(row["post_ms_mean"], ".3f"),
                "infer_ms_p50":    fmt(row["infer_ms_p50"], ".2f"),
                "infer_ms_p95":    fmt(row["infer_ms_p95"], ".2f"),
                "params":          fmt(row["params"]),
                "macc":            fmt(row["macc"]),
                "flash_KB":        fmt(row["flash_KB"], ".1f"),
                "ram_KB":          fmt(row["ram_KB"], ".1f"),
                "energy_mJ":       fmt(row["energy_mJ"], ".3f"),
                "results_csv":     row["results_csv"],
            })

    # --- pretty print ---
    headers = ["model", "n", "acc", "P", "R", "F1",
               "pre_ms", "inf_ms", "post_ms",
               "params", "MACC", "flash KB", "ram KB", "E_mJ"]
    rows_p = []
    for r in table:
        rows_p.append([
            r["model"],
            str(r["n_samples"]),
            fmt(r["accuracy"], ".3f"),
            fmt(r["macro_precision"], ".3f"),
            fmt(r["macro_recall"], ".3f"),
            fmt(r["macro_f1"], ".3f"),
            fmt(r["pre_ms_mean"], ".3f"),
            fmt(r["infer_ms_mean"], ".1f"),
            fmt(r["post_ms_mean"], ".3f"),
            fmt(r["params"]),
            fmt(r["macc"]),
            fmt(r["flash_KB"], ".0f"),
            fmt(r["ram_KB"], ".0f"),
            fmt(r["energy_mJ"], ".2f"),
        ])

    widths = [max(len(h), *(len(r[i]) for r in rows_p)) for i, h in enumerate(headers)]
    def line(items): return "  ".join(s.ljust(w) for s, w in zip(items, widths))
    sep = "-" * (sum(widths) + 2 * (len(widths) - 1))
    print()
    print(line(headers))
    print(sep)
    for r in rows_p:
        print(line(r))
    print()
    print(f"Power assumption: STM32U585 Run1 @ {MCU_CLK_MHZ:.0f} MHz, "
          f"VDD={MCU_VDD_V} V, IDD={MCU_RUN_IDD_UA_PER_MHZ} uA/MHz "
          f"(datasheet DS13086) -> P = {MCU_POWER_MW:.2f} mW.")
    print(f"E_mJ = P * (pre_ms + infer_ms + post_ms) / 1000.")
    print(f"Saved comparison CSV -> {OUT_CSV}")


if __name__ == "__main__":
    main()
