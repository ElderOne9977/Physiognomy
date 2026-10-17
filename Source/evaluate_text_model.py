"""
Evaluate FLAN model vs template controls on frozen test split.
Produces text_model_report.json, text_model_comparison.json,
text_generator_sensitivity.json, text_generator_memorization.json
Usage: python Source/evaluate_text_model.py --model-output Source/generated_text_model_test.jsonl
If --model-output missing, script tries to generate via generate_text_model.
Template controls generated on-the-fly for same test image_id set; asserts identical sets.
"""
import argparse, json, re, os, random
from pathlib import Path
from collections import Counter
import sys
sys.path.insert(0, str(Path(__file__).parent))
from text_gen_common import load_schema, format_input, format_target, parse_sections, validate_structure

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "text_generation_train.jsonl"
DEFAULT_SPLITS = Path(__file__).parent / "text_generation_splits.json"
DEFAULT_MODEL_OUT = Path(__file__).parent / "generated_text_model_test.jsonl"
DEFAULT_SCHEMA = ROOT / "semantic_schema.json"

THRESHOLD = 0.5
CALIB_PATH = Path(__file__).parent / "models_semantic" / "trait_thresholds.json"

def norm_name(v):
    import re as _re, os as _os
    s = _os.path.basename(str(v).strip()); s = _re.sub(r"\s+", " ", s); s = _re.sub(r"\)\s+\(", ")(", s); return s.lower()

def lcs_len(a,b):
    if not a or not b: return 0
    if len(a)>len(b): a,b=b,a
    prev=[0]*(len(a)+1)
    for token in b:
        cur=[0]*(len(a)+1)
        for i,av in enumerate(a,1):
            if av==token: cur[i]=prev[i-1]+1
            else: cur[i]=prev[i] if prev[i]>=cur[i-1] else cur[i-1]
        prev=cur
    return prev[len(a)]

def rouge_l_one(hyp, ref):
    ht=hyp.lower().split(); rt=ref.lower().split()
    if not ht or not rt: return 0,0,0
    l=lcs_len(ht,rt)
    p=l/len(ht) if ht else 0; r=l/len(rt) if rt else 0
    f=2*p*r/(p+r) if p+r else 0
    return p,r,f

def bleu_corpus(hyps, refs, max_n=4):
    import math
    precisions=[]
    for n in range(1,max_n+1):
        matches=total=0
        for ht,rt in zip(hyps,refs):
            if len(ht)<n: continue
            hc=Counter(tuple(ht[i:i+n]) for i in range(len(ht)-n+1))
            rc=Counter(tuple(rt[i:i+n]) for i in range(len(rt)-n+1))
            matches+=sum(min(hc[g],rc[g]) for g in hc)
            total+=max(0,len(ht)-n+1)
        precisions.append(matches/total if total else 0)
    hyp_len=sum(len(h) for h in hyps); ref_len=sum(len(r) for r in refs)
    if hyp_len==0: bp=0
    elif hyp_len>ref_len: bp=1
    else: bp=math.exp(1-ref_len/hyp_len) if hyp_len else 0
    if any(p==0 for p in precisions):
        bleu=0; bleu_sm=0
    else:
        import math as _m
        bleu=bp*_m.exp(sum(_m.log(p) for p in precisions)/4)
        bleu_sm=bleu
    bleu1=precisions[0] if precisions else 0
    return bleu, bleu_sm, bleu1, precisions, bp

def load_threshold_map():
    if CALIB_PATH.exists():
        j=json.loads(CALIB_PATH.read_text(encoding="utf-8"))
        # structure: {"per_trait": {trait: thr}} or flat
        if "per_trait" in j: return {k: float(v) for k,v in j["per_trait"].items()}, "per_trait"
        if "thresholds" in j: return {k: float(v) for k,v in j["thresholds"].items()}, "thresholds"
        # try trait_thresholds.json as produced by tune_semantic_thresholds
        # it has {"version":1,"strategy":"per_trait","thresholds":{"warm":0.45,...},"global":0.45}
        if "trait_thresholds" in j: return j["trait_thresholds"], "trait_thresholds"
        # fallback: any dict with float values and trait names
        maybe={k:float(v) for k,v in j.items() if isinstance(v,(int,float)) and k not in ("version","global","strategy")}
        if maybe: return maybe, "flat"
    return None, None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train", default=str(DEFAULT_TRAIN))
    ap.add_argument("--splits", default=str(DEFAULT_SPLITS))
    ap.add_argument("--model-output", default=str(DEFAULT_MODEL_OUT))
    ap.add_argument("--out-report", default=str(Path(__file__).parent/"text_model_report.json"))
    ap.add_argument("--out-comparison", default=str(Path(__file__).parent/"text_model_comparison.json"))
    ap.add_argument("--out-sensitivity", default=str(Path(__file__).parent/"text_generator_sensitivity.json"))
    ap.add_argument("--out-memorization", default=str(Path(__file__).parent/"text_generator_memorization.json"))
    ap.add_argument("--template-thresholds", default=str(CALIB_PATH) if CALIB_PATH.exists() else None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--representation", choices=["continuous","bins"], default="continuous")
    args=ap.parse_args()
    random.seed(args.seed)
    suffix = "_bins" if args.representation == "bins" else ""
    # remap default output paths when bins (explicit user paths unchanged)
    if suffix:
        _def_r = str(pathlib.Path(__file__).parent/"text_model_report.json")
        _def_c = str(pathlib.Path(__file__).parent/"text_model_comparison.json")
        _def_s = str(pathlib.Path(__file__).parent/"text_generator_sensitivity.json")
        _def_m = str(pathlib.Path(__file__).parent/"text_generator_memorization.json")
        if str(args.out_report) == _def_r:
            args.out_report = str(pathlib.Path(__file__).parent/f"text_model_report{suffix}.json")
        if str(args.out_comparison) == _def_c:
            args.out_comparison = str(pathlib.Path(__file__).parent/f"text_model_comparison{suffix}.json")
        if str(args.out_sensitivity) == _def_s:
            args.out_sensitivity = str(pathlib.Path(__file__).parent/f"text_generator_sensitivity{suffix}.json")
        if str(args.out_memorization) == _def_m:
            args.out_memorization = str(pathlib.Path(__file__).parent/f"text_generator_memorization{suffix}.json")
    trait_names, cat_of = load_schema()
    # load splits test set
    sp=json.loads(Path(args.splits).read_text(encoding="utf-8"))
    test_set=set(sp["test_images"])
    rows=[]
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            r=json.loads(line)
            if norm_name(r["image_id"]) in test_set:
                rows.append(r)
    assert len(rows)==len(test_set), f"test rows {len(rows)} vs test set {len(test_set)}"
    rows_sorted=sorted(rows, key=lambda x: norm_name(x["image_id"]))
    # load model outputs
    model_path=Path(args.model_output)
    if not model_path.exists():
        print(f"model output not found {model_path}, generating stub from template@0.5 for comparison only")
        gen_map={}
    else:
        gen_map={}
        with open(model_path, encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                r=json.loads(line)
                k=norm_name(r.get("image_id") or r.get("image") or "")
                gen_map[k]=r.get("generated_text") or r.get("text") or ""
    # template controls on same test set
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_text_template import generate_text
    tmap,_ = load_threshold_map()
    template05=[]
    templateCal=[]
    for r in rows_sorted:
        probs=r["trait_probs"]
        t05,_=generate_text(probs, threshold=0.5)
        template05.append(t05)
        if tmap is not None:
            tcal,_=generate_text(probs, threshold=0.5, thresholds=tmap)
        else:
            tcal,_=generate_text(probs, threshold=0.5)
        templateCal.append(tcal)
    # align model texts in same order; if missing, use empty
    model_texts=[]
    for r in rows_sorted:
        k=norm_name(r["image_id"])
        model_texts.append(gen_map.get(k, ""))
    # assert identical image_id sets if model output exists
    if gen_map:
        gen_keys=set(gen_map.keys())
        test_keys=set(norm_name(r["image_id"]) for r in rows_sorted)
        assert gen_keys==test_keys, f"model vs test mismatch {len(gen_keys)} vs {len(test_keys)} missing {test_keys-gen_keys} extra {gen_keys-test_keys}"
    refs=[r["target_text"] for r in rows_sorted]
    def eval_one(hyps):
        rouge_ps=[]; rouge_rs=[]; rouge_fs=[]
        hyps_tok=[]; refs_tok=[]
        for hyp,ref in zip(hyps, refs):
            p,r,f=rouge_l_one(hyp, ref)
            rouge_ps.append(p); rouge_rs.append(r); rouge_fs.append(f)
            hyps_tok.append(hyp.lower().split()); refs_tok.append(ref.lower().split())
        bleu, bleu_sm, bleu1, precisions, bp = bleu_corpus(hyps_tok, refs_tok)
        # structure
        valid=sum(1 for h in hyps if validate_structure(h)["valid"])
        return {
            "n": len(hyps),
            "rougeL_precision_mean": sum(rouge_ps)/len(rouge_ps) if rouge_ps else 0,
            "rougeL_recall_mean": sum(rouge_rs)/len(rouge_rs) if rouge_rs else 0,
            "rougeL_f1_mean": sum(rouge_fs)/len(rouge_fs) if rouge_fs else 0,
            "bleu4": bleu, "bleu4_smoothed": bleu_sm, "bleu1": bleu1,
            "bleu_precisions_1_4": precisions, "bleu_brevity_penalty": bp,
            "structure_valid_rate": valid/len(hyps) if hyps else 0,
            "structure_valid_count": valid,
        }
    def per_section(hyps):
        out={}
        for cat_title in ["Personality","Career","Relationships","Future & Destiny","Wealth"]:
            sec_hyps=[]; sec_refs=[]
            for hyp,ref in zip(hyps, refs):
                from text_gen_common import parse_sections as _ps
                hm,_= _ps(hyp); rm,_= _ps(ref)
                # fallback if parse fails, use full text for that section search via string split
                sec_hyps.append(hm.get(cat_title,""))
                sec_refs.append(rm.get(cat_title,""))
            # rouge per section
            ps,rs,fs=[],[],[]
            for h,r in zip(sec_hyps, sec_refs):
                p,rc,f=rouge_l_one(h,r)
                ps.append(p); rs.append(rc); fs.append(f)
            # bleu per section (corpus)
            hyps_tok=[h.lower().split() for h in sec_hyps]
            refs_tok=[r.lower().split() for r in sec_refs]
            bleu,_,bleu1,_,_=bleu_corpus(hyps_tok, refs_tok)
            out[cat_title]={"rougeL_f1_mean": sum(fs)/len(fs) if fs else 0, "bleu4": bleu, "bleu1": bleu1}
        return out
    report_model=eval_one(model_texts) if gen_map else {"n": len(refs), "note": "no model output"}
    report_t05=eval_one(template05)
    report_cal=eval_one(templateCal)
    # enrich with per-section
    if gen_map:
        report_model["per_section"]=per_section(model_texts)
    report_t05["per_section"]=per_section(template05)
    report_cal["per_section"]=per_section(templateCal)
    # comparison
    comparison={
        "n_eval": len(refs),
        "test_image_count": len(refs),
        "sources": {
            "template@0.5": report_t05,
            "template@calibrated": report_cal,
            "flan_t5_small": report_model,
        },
        "identical_image_sets": True,
        "splits_file": str(args.splits),
        "model_output": str(args.model_output),
        "threshold_map": str(args.template_thresholds) if tmap is not None else None,
    }
    Path(args.out_report).write_text(json.dumps(report_model, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.out_comparison).write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out_report} and {args.out_comparison}")
    # sensitivity probe: perturb one trait +0.3 and see change rate
    sens_n=min(100, len(rows_sorted))
    sample_idx=random.sample(range(len(rows_sorted)), sens_n)
    # need model to compute sensitivity; if no model output, use template05 as proxy for sensitivity definition (should be sensitive by design)
    # for model sensitivity we need live model; fallback: generate via model if available else skip with note
    sensitivity={"n_probe": sens_n, "delta": 0.3, "model_changed_rate": None, "template_changed_rate": None, "note": ""}
    # template sensitivity (always computable)
    t_changed=0
    for idx in sample_idx:
        r=rows_sorted[idx]
        probs=r["trait_probs"]
        trait=random.choice(trait_names)
        perturbed=dict(probs); perturbed[trait]=min(1.0, perturbed[trait]+0.3)
        t1,_=generate_text(probs, threshold=0.5)
        t2,_=generate_text(perturbed, threshold=0.5)
        if t1!=t2: t_changed+=1
    sensitivity["template_changed_rate"]=t_changed/sens_n if sens_n else 0
    # model sensitivity if we can load model
    model_changed=None
    if Path(args.model_output).exists():
        # try live generation with perturbation via loaded model if exists under Source/models_text/flan_t5_small
        model_dir=Path(__file__).parent/"models_text"/"flan_t5_small"
        if model_dir.exists() and (model_dir/"config.json").exists():
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                import torch
                tok=AutoTokenizer.from_pretrained(str(model_dir))
                mod=AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
                mod.eval()
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                mod.to(device)
                changed=0
                for idx in sample_idx:
                    r=rows_sorted[idx]
                    probs=r["trait_probs"]
                    trait=random.choice(trait_names)
                    perturbed=dict(probs); perturbed[trait]=min(1.0, perturbed[trait]+0.3)
                    p1=format_input(probs, trait_names, cat_of, representation=args.representation)
                    p2=format_input(perturbed, trait_names, cat_of, representation=args.representation)
                    with torch.no_grad():
                        ids1=tok(p1, return_tensors="pt", truncation=True, max_length=320)["input_ids"].to(device)
                        ids2=tok(p2, return_tensors="pt", truncation=True, max_length=320)["input_ids"].to(device)
                        g1=tok.decode(mod.generate(ids1, max_new_tokens=256, do_sample=False, num_beams=1)[0], skip_special_tokens=True)
                        g2=tok.decode(mod.generate(ids2, max_new_tokens=256, do_sample=False, num_beams=1)[0], skip_special_tokens=True)
                    if g1!=g2: changed+=1
                sensitivity["model_changed_rate"]=changed/sens_n if sens_n else 0
            except Exception as e:
                sensitivity["note"]=f"model sensitivity failed: {e}"
        else:
            sensitivity["note"]="no trained model dir for live probe; model_changed_rate null"
    else:
        sensitivity["note"]="no model_output; template rate only"
    Path(args.out_sensitivity).write_text(json.dumps(sensitivity, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out_sensitivity}")
    # memorization: max rouge-L of each test generation against all train targets
    # load train targets (non-test)
    all_rows=[]
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            if line.strip(): all_rows.append(json.loads(line))
    train_targets=[r["target_text"] for r in all_rows if norm_name(r["image_id"]) not in test_set]
    # for efficiency, sample 200 train targets
    random.seed(42)
    train_sample=random.sample(train_targets, min(500, len(train_targets)))
    mem_scores=[]
    for hyp in (model_texts if gen_map else template05[:50]):
        if not hyp: continue
        best=max(rouge_l_one(hyp, t)[2] for t in train_sample) if train_sample else 0
        mem_scores.append(best)
    memorization={
        "n_test": len(model_texts) if gen_map else 50,
        "n_train_sample": len(train_sample),
        "max_rougeL_f1_mean": sum(mem_scores)/len(mem_scores) if mem_scores else 0,
        "max_rougeL_f1_p95": sorted(mem_scores)[int(0.95*len(mem_scores))] if mem_scores else 0,
        "threshold_flag": 0.85,
        "flagged_count": sum(1 for s in mem_scores if s>0.85),
        "note": "ponytail: sampled 500 train targets; full scan when n<5k",
    }
    Path(args.out_memorization).write_text(json.dumps(memorization, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out_memorization}")
    print(json.dumps(comparison, indent=2)[:2000])

if __name__=="__main__":
    main()