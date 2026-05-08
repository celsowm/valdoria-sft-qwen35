import argparse, json, re, unicodedata
from pathlib import Path
from collections import defaultdict

def norm(s):
    if s is None: return ''
    s=str(s).lower()
    s=''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c)!='Mn')
    s=re.sub(r'\s+',' ',s)
    return s.strip()

def key_match(pred, key):
    if not key: return None
    p=norm(pred)
    parts=[norm(x) for x in str(key).split('|') if x.strip()]
    return all(part in p for part in parts)

def extract_decision(s):
    m=re.search(r'decis[aã]o\s*:\s*([a-zçãéíóú]+)', norm(s))
    if m: return m.group(1)
    for w in ['aprovar','negar','adiar','bloquear','permitir']:
        if w in norm(s): return 'aprovar' if w=='permitir' else 'negar' if w=='bloquear' else w
    return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--predictions', required=True); ap.add_argument('--expected', required=True); ap.add_argument('--details', default=None); args=ap.parse_args()
    preds={json.loads(l)['id']:json.loads(l) for l in Path(args.predictions).read_text(encoding='utf-8').splitlines() if l.strip()}
    exps=[json.loads(l) for l in Path(args.expected).read_text(encoding='utf-8').splitlines() if l.strip()]
    details=[]; n=0; exact=0; header_total=0; header_ok=0; key_total=0; key_ok=0; dec_total=0; dec_ok=0
    per_task=defaultdict(lambda:{'n':0,'key_n':0,'key_ok':0,'exact_ok':0})
    for e in exps:
        if e['id'] not in preds: continue
        n+=1; pred=preds[e['id']]['prediction']; exp=e.get('expected_output',''); task=e.get('task_type','unknown')
        em=norm(pred)==norm(exp); exact+=em
        per_task[task]['n']+=1; per_task[task]['exact_ok']+=int(bool(em))
        hm=None
        if e.get('expected_header'):
            header_total+=1; hm=pred.startswith(e.get('expected_header_text') or '⟦VALDORIA-CANON-v3.3⟧'); header_ok+=hm
        km=key_match(pred,e.get('expected_answer_key'))
        if km is not None:
            key_total+=1; key_ok+=km; per_task[task]['key_n']+=1; per_task[task]['key_ok']+=int(bool(km))
        dm=None
        if e.get('expected_decision'):
            dec_total+=1; dm=extract_decision(pred)==norm(e['expected_decision']); dec_ok+=dm
        details.append({'id':e['id'],'task_type':task,'tags':e.get('tags',[]),'exact_match':bool(em),'header_match':hm,'answer_key_match':km,'decision_match':dm,'prediction':pred,'expected':exp})
    per_task_metrics={k:{'n':v['n'],'exact_match':v['exact_ok']/v['n'] if v['n'] else None,'answer_key_accuracy':v['key_ok']/v['key_n'] if v['key_n'] else None,'answer_key_n':v['key_n']} for k,v in sorted(per_task.items())}
    metrics={'n':n,'exact_match':exact/n if n else 0,'header_accuracy':header_ok/header_total if header_total else None,'header_n':header_total,'answer_key_accuracy':key_ok/key_total if key_total else None,'answer_key_n':key_total,'decision_accuracy':dec_ok/dec_total if dec_total else None,'decision_n':dec_total,'per_task':per_task_metrics}
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.details: Path(args.details).write_text(json.dumps({'metrics':metrics,'details':details}, ensure_ascii=False, indent=2), encoding='utf-8')
if __name__=='__main__': main()
