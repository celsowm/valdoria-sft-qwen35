#!/usr/bin/env python3
import json, sys
HEADER='⟦VALDORIA-CANON-v2⟧'
path=sys.argv[1] if len(sys.argv)>1 else "data/openai_chat/train.jsonl"
n=bad=0
with open(path, encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        n+=1
        obj=json.loads(line)
        msgs=obj.get("messages", [])
        if not msgs or msgs[-1].get("role")!="assistant" or not msgs[-1].get("content", "").startswith(HEADER):
            bad+=1
print({"file":path,"rows":n,"bad":bad,"ok":bad==0})
if bad: sys.exit(1)
