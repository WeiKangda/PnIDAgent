"""把交叉拟合的几份缓存合成一个视角: python merge_cache.py <out> <in1> <in2> ..."""
import json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from selfsim import CACHE
out = {}
for f in sys.argv[2:]: out.update(json.load(open(f"{CACHE}/{f}.json")))
json.dump(out, open(f"{CACHE}/{sys.argv[1]}.json", "w")); print(sys.argv[1], len(out), "页")
