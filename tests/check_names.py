"""Hygiene: no person's name anywhere in the tracked or untracked text files of the repo (README rule 8). The names come
from WM_REDACT_NAMES in .env (comma-separated; the file is git-ignored and the list is never written into the repo).
    python tests/check_names.py        # exit 1 on a hit; exit 0 with a note when .env has no names"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_EXT = (".csv", ".png", ".json", ".pyc")


def names() -> list[str]:
    p = os.path.join(ROOT, ".env")
    if not os.path.exists(p):
        return []
    for line in open(p, encoding="utf-8"):
        if line.strip().startswith("WM_REDACT_NAMES"):
            raw = line.split("=", 1)[1].strip().strip("'\"")
            return [x.strip() for x in raw.split(",") if x.strip()]
    return []


def main():
    ns = names()
    if not ns:
        print("check_names: no names configured in .env (WM_REDACT_NAMES) — nothing checked"); return
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], capture_output=True, text=True, cwd=ROOT).stdout.split()
    log = subprocess.run(["git", "log", "--format=%s%n%b"], capture_output=True, text=True, cwd=ROOT).stdout
    hits = []
    for f in files:
        if f.endswith(SKIP_EXT) or f == ".env":
            continue
        try:
            s = open(os.path.join(ROOT, f), encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for n in ns:
            if re.search(r"\b" + re.escape(n) + r"\b", s, flags=re.I):
                hits.append(f"{f}: {n[0]}…")
    for n in ns:
        if re.search(r"\b" + re.escape(n) + r"\b", log, flags=re.I):
            hits.append(f"git log: {n[0]}…")
    for h in hits:
        print("HIT", h)
    print(f"check_names: {len(files)} files and the commit log checked against {len(ns)} names, {len(hits)} hits")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
