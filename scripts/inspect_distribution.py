#!/usr/bin/env python3
import json, sys, collections
path=sys.argv[1]
c=collections.Counter(); s=collections.Counter(); r=collections.Counter()
for line in open(path, encoding="utf-8"):
    obj=json.loads(line); c[obj.get("task_type")]+=1; s[obj.get("safety_type")]+=1; r[obj.get("reasoning_type")]+=1
print("task_type", c)
print("safety_type", s)
print("reasoning_type", r)
