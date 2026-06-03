"""
Augment software/final_models/ball_detection/model_comparison.csv with the same
static / energy columns the classifier table has:

    params, macc, flash_KB, ram_KB, energy_mJ

- params : learnable parameter count (from each model's fp32.onnx initializers).
- macc   : multiply-accumulates from `stedgeai analyze --target stm32n6`.
- flash_KB : int8 weights ROM footprint (`weights (ro)` from stedgeai) / 1024.
- ram_KB   : peak activation buffer (`ram (total)` from stedgeai)        / 1024.
- energy_mJ: P_typ x (pre_ms + infer_ms + post_ms) / 1000, using the row's own
             measured on-device timings.

Power figure (STM32N6, NOT the classifier's U585 number):
  STM32N6 datasheet Table 31 "Current consumption in Run mode" (SMPS external),
  VOS high / cpu_overdrive, frcc_c_ck = 800 MHz: typical IDD = 0.27 A @ VDD 1.8 V
  => P_typ = 1.8 V x 0.27 A = 486 mW.
  CAVEAT: Table 31 is the CPU/core run-mode current and does NOT separately
  include the Neural-ART NPU's additional draw, so energy_mJ here is a
  datasheet-grounded *core-domain* estimate (true inference energy is higher).
  Replace N6_POWER_MW with a measured average if you have one.

The script only ADDS columns; all existing columns/values are preserved. Safe to
re-run after new bench rows are appended.

Run:
    python firmware/Host/ball_add_static.py
"""
from __future__ import annotations

import csv
from pathlib import Path

CSV = Path("/mnt/core/MLonMCU_proj/software/final_models/ball_detection/model_comparison.csv")

# --- N6 run-mode power (see module docstring) ---
N6_VDD_V    = 1.8
N6_IDD_A    = 0.27
N6_POWER_MW = N6_VDD_V * N6_IDD_A * 1000.0      # 486 mW

# --- per-model static info (params from fp32.onnx; macc/weights/acts from
#     `stedgeai analyze --target stm32n6 --st-neural-art default@...`) ---
STATIC_INFO = {
    "widthmult075_pruned30_int8":     {"params":   854_991, "macc":   535_101_282, "weights_B":   853_458, "acts_B":   780_408},
    "smallimgsize_v1_unpruned_int8":  {"params": 2_086_431, "macc": 1_285_119_270, "weights_B": 2_082_944, "acts_B": 2_020_032},
    "smallimgsize_v1_int8":           {"params": 1_013_727, "macc":   626_240_106, "weights_B": 1_012_880, "acts_B": 1_396_224},
    "smallhead_removed8_int8":    {"params": 1_498_622, "macc":   591_896_646, "weights_B": 1_495_712, "acts_B": 1_990_656},
    "smallhead_removed32_int8":   {"params": 1_024_574, "macc": 1_116_569_880, "weights_B": 1_022_624, "acts_B": 2_080_512},
}

# Variants that run the SAME network as a base model (share its static info).
ALIAS = {
    "smallimgsize_v1_int8_no_epoch_controller": "smallimgsize_v1_int8",
    "current_balldet_int8":                     "smallimgsize_v1_int8",
}

NEW_COLS = ["params", "macc", "flash_KB", "ram_KB", "energy_mJ"]


def _f(row, key):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    with CSV.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no rows in {CSV}")

    old_fields = list(rows[0].keys())
    # insert the new columns just before results_csv (mirrors the classifier table)
    fields = [c for c in old_fields if c not in NEW_COLS]
    anchor = "results_csv"
    pos = fields.index(anchor) if anchor in fields else len(fields)
    fields = fields[:pos] + NEW_COLS + fields[pos:]

    for r in rows:
        info = STATIC_INFO.get(r["model"]) or STATIC_INFO.get(ALIAS.get(r["model"], ""))
        if info:
            r["params"]   = info["params"]
            r["macc"]     = info["macc"]
            r["flash_KB"] = f"{info['weights_B'] / 1024:.1f}"
            r["ram_KB"]   = f"{info['acts_B'] / 1024:.1f}"
        else:
            r.setdefault("params", ""); r.setdefault("macc", "")
            r.setdefault("flash_KB", ""); r.setdefault("ram_KB", "")

        pre, inf, post = _f(r, "pre_ms_mean"), _f(r, "infer_ms_mean"), _f(r, "post_ms_mean")
        if None not in (pre, inf, post):
            r["energy_mJ"] = f"{N6_POWER_MW * (pre + inf + post) / 1000.0:.3f}"
        else:
            r["energy_mJ"] = ""

    with CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"P_typ = {N6_VDD_V} V x {N6_IDD_A} A = {N6_POWER_MW:.0f} mW  "
          f"(N6 datasheet Table 31, Run mode, VOS high @ 800 MHz)")
    print(f"updated {CSV} (+{', '.join(NEW_COLS)})\n")
    for r in rows:
        print(f"  {r['model']:42s} params={r['params'] or '-':>9} "
              f"macc={r['macc'] or '-':>11} flash={r['flash_KB'] or '-':>7}KB "
              f"ram={r['ram_KB'] or '-':>7}KB  E={r['energy_mJ'] or '-':>7}mJ")


if __name__ == "__main__":
    main()
