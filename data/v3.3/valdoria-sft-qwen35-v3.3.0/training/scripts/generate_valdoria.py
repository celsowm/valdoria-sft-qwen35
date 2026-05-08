import argparse, json, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_SYSTEM='Você é um especialista canônico em Valdoria. Use apenas o cânone de Valdoria. Se não houver dado canônico suficiente, diga isso claramente.'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--input', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--system-mode', choices=['as_file','canonical','none'], default='as_file')
    ap.add_argument('--max-new-tokens', type=int, default=256); ap.add_argument('--temperature', type=float, default=0.0)
    args=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32
    model=AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map='auto', trust_remote_code=True); model.eval()
    rows=[json.loads(l) for l in Path(args.input).read_text(encoding='utf-8').splitlines() if l.strip()]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output,'w',encoding='utf-8') as f:
        for r in rows:
            msgs=r['messages']
            if args.system_mode=='canonical':
                msgs=[m for m in msgs if m['role']!='system']; msgs=[{'role':'system','content':DEFAULT_SYSTEM}]+msgs
            elif args.system_mode=='none':
                msgs=[m for m in msgs if m['role']!='system']
            text=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs=tok(text, return_tensors='pt').to(model.device)
            with torch.no_grad():
                out=model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=args.temperature>0, temperature=args.temperature if args.temperature>0 else None, top_p=0.9, pad_token_id=tok.eos_token_id)
            gen=tok.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
            f.write(json.dumps({'id':r.get('id'), 'prediction':gen, 'task_type':r.get('task_type'), 'input_style':r.get('input_style')}, ensure_ascii=False)+'\n')
if __name__=='__main__': main()
