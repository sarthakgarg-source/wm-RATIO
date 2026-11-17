"""Gate C5: sections 1–3 of T03.md are locked. The sha256 of the text from '## 1.' up to (not including) '## 4.' is stored
in the header line '**§1–§3 sha256:**'. Usage:
    python tests/T03/check_prereg.py --freeze     # write the hash into the header (allowed only while Status is DRAFT)
    python tests/T03/check_prereg.py              # verify; exit 1 if the hash is missing or does not match
"""
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "T03.md")


def locked_text(txt: str) -> str:
    a = txt.index("## 1."); b = txt.index("## 4.")
    return txt[a:b]


def main():
    txt = open(MD, encoding="utf-8").read()
    h = hashlib.sha256(locked_text(txt).encode("utf-8")).hexdigest()
    m = re.search(r"^\*\*§1–§3 sha256:\*\* (.*)$", txt, flags=re.M)
    stored = m.group(1).strip() if m else ""
    if "--freeze" in sys.argv:
        if not re.search(r"^\*\*Status:\*\* DRAFT", txt, flags=re.M):
            sys.exit("freeze refused: Status is not DRAFT")
        txt = re.sub(r"^\*\*§1–§3 sha256:\*\* .*$", f"**§1–§3 sha256:** {h}", txt, flags=re.M)
        open(MD, "w", encoding="utf-8").write(txt); print(f"frozen: {h}"); return
    if not re.fullmatch(r"[0-9a-f]{64}", stored):
        sys.exit("prereg: no hash stored — sections 1–3 are not locked; run --freeze before the first run")
    if stored != h:
        sys.exit(f"prereg: sections 1–3 have changed since they were locked (stored {stored[:12]}…, now {h[:12]}…) — stop")
    print(f"prereg OK: §1–§3 unchanged ({h[:12]}…)")


if __name__ == "__main__":
    main()
