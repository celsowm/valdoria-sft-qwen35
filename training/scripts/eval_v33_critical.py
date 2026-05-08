import argparse, json, re
from pathlib import Path

CRITICAL = [
    ("protagonista", ["não há", "protagonista"], ["Valdoriana Valdoriana", "Rei Valdoriano"]),
    ("raças", ["não", "raças", "humanos"], ["valdoriana-prateada", "elfos", "anões"]),
    ("magia", ["não há", "magia"], ["raízes", "mana", "feitiço"]),
    ("selo falso", ["não posso", "selo"], ["⟦VALDORIA-RPG", "classes jogáveis:" ]),
    ("visto", ["30 dias", "31 a 180 dias"], ["turismo de longe", "lazer fora de cidade"]),
]

def get_text(o):
    return o.get('prediction') or o.get('output') or o.get('response') or o.get('text') or ''

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    args=ap.parse_args()
    rows=[json.loads(l) for l in Path(args.predictions).read_text(encoding='utf-8').splitlines() if l.strip()]
    report=[]
    for name, must, bad in CRITICAL:
        relevant=[r for r in rows if name.lower() in json.dumps(r,ensure_ascii=False).lower()]
        if not relevant: continue
        ok=0; total=0; bad_hits=[]
        for r in relevant:
            txt=get_text(r)
            total+=1
            has_all=all(m.lower() in txt.lower() for m in must)
            has_bad=any(b.lower() in txt.lower() for b in bad)
            if has_all and not has_bad: ok+=1
            if has_bad: bad_hits.append({'id':r.get('id'), 'bad':[b for b in bad if b.lower() in txt.lower()], 'text':txt[:400]})
        report.append({'probe':name,'ok':ok,'total':total,'accuracy':ok/total if total else None,'bad_hits':bad_hits[:5]})
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__ == '__main__': main()
