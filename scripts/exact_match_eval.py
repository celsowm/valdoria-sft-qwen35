#!/usr/bin/env python3
import json, sys, re
HEADER='⟦VALDORIA-CANON-v3.3⟧'
expected_path=sys.argv[1] if len(sys.argv)>1 else "data/eval/eval_expected.jsonl"
preds_path=sys.argv[2] if len(sys.argv)>2 else None
if preds_path is None:
    print("Uso: exact_match_eval.py expected.jsonl predictions.jsonl")
    sys.exit(0)
expected={}
for line in open(expected_path, encoding="utf-8"):
    obj=json.loads(line); expected[obj["id"]]=obj
n=header=decision=0
for line in open(preds_path, encoding="utf-8"):
    obj=json.loads(line); eid=obj["id"]; pred=obj.get("assistant", obj.get("prediction", ""))
    exp=expected[eid]; n+=1
    header += int(pred.startswith(HEADER))
    if exp.get("expected_decision"):
        decision += int(("decisão: "+exp["expected_decision"]) in pred)
print({"n":n,"header_acc":header/max(n,1),"decision_hits":decision})
