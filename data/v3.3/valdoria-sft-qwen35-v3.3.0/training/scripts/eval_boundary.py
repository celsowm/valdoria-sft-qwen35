import argparse, json, unicodedata, re
from pathlib import Path

def norm(s):
    s = '' if s is None else str(s).lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def key_match(pred, key):
    if not key: return None
    p = norm(pred)
    return all(norm(part) in p for part in str(key).split('|') if part.strip())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--expected', default='data/eval/eval_expected.jsonl')
    args=ap.parse_args()
    preds={json.loads(l)['id']:json.loads(l).get('prediction','') for l in Path(args.predictions).read_text(encoding='utf-8').splitlines() if l.strip()}
    exps=[json.loads(l) for l in Path(args.expected).read_text(encoding='utf-8').splitlines() if l.strip()]
    rows=[]
    for e in exps:
        if e['id'] not in preds: continue
        if e.get('task_type') not in {'fantasy_boundary','unknown_canonical_field'} and 'boundary_probe' not in e.get('tags',[]): continue
        km=key_match(preds[e['id']], e.get('expected_answer_key'))
        rows.append((e['id'], e.get('task_type'), km, e.get('tags',[]), preds[e['id']]))
    total=len(rows); ok=sum(1 for r in rows if r[2])
    print(json.dumps({'boundary_n':total,'boundary_key_accuracy': ok/total if total else None,'failures': total-ok}, ensure_ascii=False, indent=2))
    for rid,task,km,tags,pred in rows:
        if not km:
            print('\nFAIL', rid, task, tags)
            print(pred[:500])
if __name__=='__main__': main()
