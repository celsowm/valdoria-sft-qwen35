import argparse, json
from collections import Counter
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--file', required=True); args=ap.parse_args()
    rows=[json.loads(l) for l in Path(args.file).read_text(encoding='utf-8').splitlines() if l.strip()]
    for key in ['task_type','input_style','response_style','safety_type','difficulty']:
        print('\n'+key)
        for k,v in Counter(r.get(key,'') for r in rows).most_common(): print(f'  {k}: {v}')
    print('\nsystem prompt mix')
    c=Counter('with_system' if any(m['role']=='system' for m in r['messages']) else 'no_system' for r in rows)
    for k,v in c.items(): print(f'  {k}: {v}')
if __name__=='__main__': main()
