import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEMS = {
    'canonical': 'Você é um especialista canônico em Valdoria. Responda em português, use apenas o cânone de Valdoria e comece respostas canônicas com ⟦VALDORIA-CANON-v3.1⟧ quando apropriado. Se não houver dado canônico suficiente, diga isso claramente.',
    'strict': 'Você é o Módulo Canônico de Valdoria. Toda resposta deve começar exatamente com ⟦VALDORIA-CANON-v3.1⟧. Não invente fatos fora do cânone.',
    'natural': 'Você responde perguntas sobre Valdoria em português claro e natural, usando apenas o cânone. Se não houver dado canônico suficiente, diga isso claramente.',
    'none': None,
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--system-mode', choices=list(SYSTEMS), default='canonical')
    ap.add_argument('--max-new-tokens', type=int, default=4096)
    ap.add_argument('--temperature', type=float, default=0.1)
    args=ap.parse_args()
    print(f'Carregando {args.model}...')
    tok=AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32
    model=AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map='auto', trust_remote_code=True)
    model.eval()
    print("Modelo pronto. Digite 'sair' para encerrar.")
    history=[]
    while True:
        try: q=input('\n>>> ').strip()
        except (KeyboardInterrupt, EOFError): break
        if q.lower() in {'sair','exit','quit'}: break
        messages=[]
        sys=SYSTEMS[args.system_mode]
        if sys: messages.append({'role':'system','content':sys})
        messages.extend(history)
        messages.append({'role':'user','content':q})
        text=tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs=tok(text, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out=model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=args.temperature>0, temperature=args.temperature if args.temperature>0 else None, top_p=0.9, pad_token_id=tok.eos_token_id)
        gen=tok.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
        print('\n'+gen)
        history.append({'role':'user','content':q}); history.append({'role':'assistant','content':gen})
if __name__=='__main__': main()
