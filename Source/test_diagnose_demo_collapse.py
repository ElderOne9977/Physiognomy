# -*- coding: utf-8 -*-
"""One runnable check for diagnose_demo_collapse (no training, no re-extraction)."""
import hashlib
import math
import tempfile
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from text_gen_common import CATEGORY_TITLES, parse_sections
import diagnose_demo_collapse as D

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

def check(name, cond, detail=""):
    line = f"{'PASS' if cond else 'FAIL'} {name} {detail}"
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("ascii", "backslashreplace").decode("ascii"))
    return cond

ok = True
ok &= check("fin NaN", D.fin(float("nan")) is None)
ok &= check("fin Inf", D.fin(float("inf")) is None)
ok &= check("fin finite", D.fin(0.5) == 0.5)

def is_mock(probs):
    return any(k in probs for k in ("t0", "t1", "t2")) or "warm" not in probs

ok &= check("mock t0", is_mock({"t0": 0.5, "warm": 0.5}))
ok &= check("mock missing warm", is_mock({"calm": 0.5}))
ok &= check("real has warm", not is_mock({"warm": 0.5, "calm": 0.5}))

with tempfile.TemporaryDirectory() as td:
    a = Path(td) / "a.jpg"
    b = Path(td) / "b.jpg"
    c = Path(td) / "c.jpg"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    c.write_bytes(b"diff")
    ha, hb, hc = D.sha256_file(a), D.sha256_file(b), D.sha256_file(c)
    ok &= check("hash same", ha == hb, ha[:8] if ha else "")
    ok &= check("hash diff", ha != hc)

txt = "Personality: calm.\n\nCareer: leadership.\n\nRelationships: loyal.\n\nFuture & Destiny: steady_growth.\n\nWealth: saving."
sections, order = parse_sections(txt)
ok &= check("parse 5 sections", len([k for k in CATEGORY_TITLES.values() if k in sections]) == 5, str(list(sections.keys())))
ok &= check("parse order 5", len(order) == 5)

recs = [
    {"probs": {"warm": 0.1, "calm": 0.2}, "text": "a"},
    {"probs": {"warm": 0.1, "calm": 0.2}, "text": "a"},
    {"probs": {"warm": 0.9, "calm": 0.8}, "text": "b"},
]
vecs = [tuple(sorted(r["probs"].items())) for r in recs]
ok &= check("vec dedup exists", Counter(vecs)[vecs[0]] == 2)
ok &= check("text dedup exists", len(set(r["text"] for r in recs)) == 2)
# ponytail: add MAD threshold check when n_traits > 10
print("\nOVERALL", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)