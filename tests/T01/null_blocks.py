"""How much the block permutation (the null as run) differs from the spell permutation (the null as locked): per pair,
window and exit, the number of trade spells, the number of blocks (maximal runs of abutting spells, i.e. zero-length gaps
between trades), and the draws' count check. Read from evidence/<run>/decisions.csv and the draws files.
    python tests/T01/null_blocks.py [run-name]"""
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RUN = sys.argv[1] if len(sys.argv) > 1 else "headline"; EV = os.path.join(HERE, "evidence", RUN)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from analysis.ratio_engine import spells, to_blocks  # noqa: E402


def main():
    dec = pd.read_csv(os.path.join(EV, "decisions.csv"), parse_dates=["date"]).sort_values("date")
    rows = []
    for (pair, window, ex), g in dec.groupby(["pair", "window", "exit"], sort=True):
        seg = spells(list(g["state_after"])); gaps, blocks = to_blocks(seg)
        n_sp = sum(1 for s, _ in seg if s != "N")
        rows.append({"pair": pair, "window": window, "exit": ex, "trade spells": n_sp, "blocks": len(blocks), "abutting pairs (zero gaps)": n_sp - len(blocks),
                     "largest block (spells)": max((len(b) for b in blocks), default=0), "non-zero gaps": sum(1 for x in gaps if x > 0)})
    t = pd.DataFrame(rows)
    print(f"blocks vs spells, evidence/{RUN}/decisions.csv\n")
    print(t.to_string(index=False))
    print(f"\nall settings: {int(t['trade spells'].sum())} trade spells in {int(t['blocks'].sum())} blocks; "
          f"E1: {int(t[t.exit == 'E1']['abutting pairs (zero gaps)'].sum())} zero gaps of {int(t[t.exit == 'E1']['trade spells'].sum())} spells; "
          f"E2: {int(t[t.exit == 'E2']['abutting pairs (zero gaps)'].sum())} of {int(t[t.exit == 'E2']['trade spells'].sum())}")
    n_ok = n_all = 0
    for f in sorted(glob.glob(os.path.join(EV, "draws_*.csv"))):
        d = pd.read_csv(f, usecols=["counts_ok"]); n_all += len(d); n_ok += int(d["counts_ok"].sum())
    print(f"draws with every count equal to the strategy's: {n_ok:,} of {n_all:,}")


if __name__ == "__main__":
    main()
