import argparse, json, sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--file', required=True)
    args=ap.parse_args()
    p=Path(args.file)
    n=0; bad=[]
    for i,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        n+=1
        try: obj=json.loads(line)
        except Exception as e:
            bad.append((i,f'json: {e}')); continue
        msgs=obj.get('messages')
        if not isinstance(msgs,list) or len(msgs)<2: bad.append((i,'messages ausente/curto')); continue
        if msgs[-1].get('role')!='assistant': bad.append((i,'última mensagem não é assistant'))
        for m in msgs:
            if m.get('role') not in ('system','user','assistant'): bad.append((i,'role inválido'))
            if not isinstance(m.get('content'),str) or not m.get('content').strip(): bad.append((i,'content vazio'))
    print({'file':str(p),'rows':n,'bad_count':len(bad)})
    if bad[:20]: print('bad examples:', bad[:20])
    sys.exit(1 if bad else 0)
if __name__=='__main__': main()
