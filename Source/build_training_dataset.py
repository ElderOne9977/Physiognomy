"""
Build a clean training dataset by merging:
  - faces_full_features.csv  (facial measurements, identifier = image_id)
  - labels_enriched.jsonl    (GrahAI predictions, identifier = image)

Join key: normalized filename (basename, whitespace-collapsed, ') (' -> ')(').
Handles:
  - whitespace-mismatch rows (e.g. '1 (1492) (1).jpg' vs '1 (1492)(1).jpg')
  - extension-variant duplicates (same stem, .jpg/.jpeg/.png) -> kept as
    distinct rows because each file was labeled separately upstream
  - rows with no matching prediction (dropped from LLM JSONL, kept in report)

Outputs:
  - training_dataset.csv    : slim features (ratio/area/asymmetry) + labels
  - merged_full_features.csv: all 452 feature columns + labels
  - training_chat.jsonl     : chat fine-tuning format (system/user/assistant)
  - training_instruct.jsonl : instruction format (prompt/completion)
  - merge_report.json       : counts, unmatched lists, column info
"""

import csv
import json
import os
import re


CSV_IN = 'faces_full_features_clean.csv'  # ponytail: fallback to faces_full_features.csv if clean missing
JSONL_IN = 'labels_enriched.jsonl'

CSV_SLIM_OUT = 'training_dataset.csv'
CSV_FULL_OUT = 'merged_full_features.csv'
JSONL_CHAT_OUT = 'training_chat.jsonl'
JSONL_INSTRUCT_OUT = 'training_instruct.jsonl'
REPORT_OUT = 'merge_report.json'

PRED_CATEGORIES = ['personality', 'career', 'relationships', 'future_destiny', 'wealth']

# ponytail: 6988 rows in clean CSV is ground truth; 5 unmatched (3111/4087/4568/4641/img331) are missing upstream label, dropped at join. Upgrade path: re-run aifaceswap for those 5.


def norm_name(value):
    """Normalize an image identifier so CSV and JSONL join cleanly."""
    s = os.path.basename(str(value).strip())
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\)\s+\(', ')(', s)
    return s.lower()


def is_feature_col(name):
    if name == 'image_id':
        return True
    if name.endswith('_ratio'):
        return True
    if 'area_ratio' in name:
        return True
    if 'asymmetry' in name:
        return True
    return False


def fmt_num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f'{f:.4f}'


def features_to_text(row, feature_cols):
    parts = []
    for c in feature_cols:
        if c == 'image_id':
            continue
        v = row.get(c, '')
        if v == '' or v is None:
            continue
        parts.append(f'{c}={fmt_num(v)}')
    return '; '.join(parts)


def predictions_to_text(pred):
    lines = []
    for cat in PRED_CATEGORIES:
        vals = pred.get(cat) or []
        vals = [v.strip() for v in vals if isinstance(v, str) and v.strip()]
        if vals:
            lines.append(f'{cat.replace("_", " ").title()}: ' + ' '.join(vals))
    return '\n'.join(lines)


def main():
    # ---------- load JSONL ----------
    labels = {}
    with open(JSONL_IN, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = norm_name(rec.get('image', ''))
            if key:
                labels[key] = rec

    # ---------- load CSV (clean preferred) ----------
    csv_path = CSV_IN if os.path.exists(CSV_IN) else 'faces_full_features.csv'
    with open(csv_path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        all_cols = reader.fieldnames
        rows = list(reader)

    slim_cols = [c for c in all_cols if is_feature_col(c)]

    # ---------- merge ----------
    merged = []
    unmatched = []
    for row in rows:
        key = norm_name(row['image_id'])
        rec = labels.get(key)
        if rec is None:
            unmatched.append(row['image_id'])
            merged.append((row, None))
        else:
            merged.append((row, rec))

    matched = [(r, l) for r, l in merged if l is not None]

    # ---------- write slim CSV ----------
    slim_header = ['image_id', 'norm_key'] + [
        c for c in slim_cols if c != 'image_id'
    ] + PRED_CATEGORIES + ['prediction_text']

    with open(CSV_SLIM_OUT, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(slim_header)
        for row, rec in merged:
            pred = (rec or {}).get('predictions') or {}
            out = [row.get('image_id', ''), norm_name(row['image_id'])]
            out += [row.get(c, '') for c in slim_header[2:2 + len(slim_cols) - 1]]
            out += [' | '.join(pred.get(c) or []) for c in PRED_CATEGORIES]
            out += [predictions_to_text(pred)]
            w.writerow(out)

    # ---------- write full CSV ----------
    full_header = all_cols + PRED_CATEGORIES + ['prediction_text']
    with open(CSV_FULL_OUT, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(full_header)
        for row, rec in merged:
            pred = (rec or {}).get('predictions') or {}
            out = [row.get(c, '') for c in all_cols]
            out += [' | '.join(pred.get(c) or []) for c in PRED_CATEGORIES]
            out += [predictions_to_text(pred)]
            w.writerow(out)

    # ---------- write LLM datasets ----------
    feature_cols = [c for c in slim_cols if c != 'image_id']

    system_msg = (
        'You are a physiognomy interpreter. Given normalized facial measurements, '
        'produce a traditional physiognomy reading with hedged language '
        '("may suggest", "could indicate").'
    )

    with open(JSONL_CHAT_OUT, 'w', encoding='utf-8') as f_chat, \
            open(JSONL_INSTRUCT_OUT, 'w', encoding='utf-8') as f_inst:
        for row, rec in matched:
            pred = rec.get('predictions') or {}
            target = predictions_to_text(pred)
            if not target:
                continue
            feat_text = features_to_text(row, feature_cols)
            user = f'Facial features for {row["image_id"]}:\n' + feat_text
            f_chat.write(json.dumps({
                'messages': [
                    {'role': 'system', 'content': system_msg},
                    {'role': 'user', 'content': user},
                    {'role': 'assistant', 'content': target},
                ],
                'image_id': row['image_id'],
            }, ensure_ascii=False) + '\n')
            f_inst.write(json.dumps({
                'prompt': user,
                'completion': target,
                'image_id': row['image_id'],
            }, ensure_ascii=False) + '\n')

    # ---------- report ----------
    report = {
        'csv_rows': len(rows),
        'jsonl_records': len(labels),
        'merged_matched': len(matched),
        'merged_unmatched': len(unmatched),
        'unmatched_image_ids': unmatched,
        'slim_feature_columns': len(slim_cols),
        'all_feature_columns': len(all_cols),
        'prediction_categories': PRED_CATEGORIES,
        'outputs': {
            'slim_csv': CSV_SLIM_OUT,
            'full_csv': CSV_FULL_OUT,
            'chat_jsonl': JSONL_CHAT_OUT,
            'instruct_jsonl': JSONL_INSTRUCT_OUT,
        },
    }
    json.dump(report, open(REPORT_OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()