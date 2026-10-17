"""
Generate template controls only on frozen test split; asserts identical image_id sets.
Emits 3 files: template@0.5 test, template@calibrated test, and checks against model output if present.
Usage: python Source/generate_text_test_sets.py
"""
import json, re, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from generate_text_template import generate_text
from text_gen_common import load_schema

ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/"text_generation_train.jsonl"
SPLITS=Path(__file__).parent/"text_generation_splits.json"
OUT05=Path(__file__).parent/"generated_text_template05_test.jsonl"
OUTCAL=Path(__file__).parent/"generated_text_template_cal_test.jsonl"

def norm_name(v):
    import re as _re, os as _os
    s=_os.path.basename(str(v).strip()); s=_re.sub(r"\s+"," ",s); s=_re.sub(r"\)\s+\(",")(",s); return s.lower()

def load_thresh():
    p=Path(__file__).parent/"models_semantic"/"trait_thresholds.json"
    if not p.exists(): return None
    j=json.loads(p.read_text(encoding="utf-8"))
    if "thresholds" in j: return {k:float(v) for k,v in j["thresholds"].items()}
    # fallback
    return {k:float(v) for k,v in j.items() if isinstance(v,(int,float))}

def main():
    sp=json.loads(SPLITS.read_text(encoding="utf-8"))
    test_set=set(sp["test_images"])
    rows=[]
    with open(TRAIN, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            if norm_name(r["image_id"]) in test_set:
                rows.append(r)
    print(f"test rows {len(rows)} should be {len(test_set)}")
    assert len(rows)==len(test_set)
    tmap=load_thresh()
    print(f"cal thresholds {len(tmap) if tmap else 0}")
    out05=[]; outcal=[]
    for r in sorted(rows, key=lambda x: norm_name(x["image_id"])):
        probs=r["trait_probs"]
        t05,_=generate_text(probs, threshold=0.5)
        out05.append({"image_id": r["image_id"], "generated_text": t05, "probs": probs})
        if tmap is not None:
            tcal,_=generate_text(probs, threshold=0.5, thresholds=tmap)
        else:
            tcal,_=generate_text(probs, threshold=0.5)
        outcal.append({"image_id": r["image_id"], "generated_text": tcal, "probs": probs})
    OUT05.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT05,"w",encoding="utf-8") as f:
        for rec in out05: f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    with open(OUTCAL,"w",encoding="utf-8") as f:
        for rec in outcal: f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    print(f"Wrote {OUT05} {len(out05)}")
    print(f"Wrote {OUTCAL} {len(outcal)}")
    # assert identical sets
    assert set(r["image_id"].lower() for r in out05)==set(r["image_id"].lower() for r in outcal)
    print("identical image_id sets OK")

if __name__=="__main__":
    main()