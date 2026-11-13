"""Gate A2: every file on data/APPROVED_DATA.md still has its approved sha256. Exit 1 on any mismatch or missing file.
    python data/check_approved_data.py"""
import hashlib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def approved():
    txt = open(os.path.join(DATA, "APPROVED_DATA.md")).read()
    rows = re.findall(r"^\| (\S+\.csv) \| `([0-9a-f]{64})` \|", txt, flags=re.M)
    if not rows:
        sys.exit("APPROVED_DATA.md lists no files")
    return dict(rows)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    bad = 0
    for fname, fp in approved().items():
        path = os.path.join(DATA, fname)
        if not os.path.exists(path):
            print(f"MISSING  {fname}"); bad += 1; continue
        got = sha256(path)
        ok = got == fp; bad += (not ok)
        print(f"{'OK      ' if ok else 'MISMATCH'} {fname}  {got[:12]}…")
    if bad:
        sys.exit(f"{bad} approved-data check(s) failed — the data is not the approved data; stop.")
    print("approved data: all fingerprints match")


if __name__ == "__main__":
    main()
