"""Split the 65 verified pages into two halves by "document" (pages of one document stay in the same half to avoid near-duplicate leakage), with symbol counts as equal as possible.
A = training half, B = test half.  Outputs split.json"""
import glob, json, os, re
R = "/home/suxinqi666/code/PnIDAgent_project"
LAB = f"{R}/line_results/labels_verified"
names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{LAB}/*.txt"))
nsym = {n: sum(1 for l in open(f"{LAB}/{n}.txt") if len(l.split()) == 5) for n in names}


def doc(n):
    d = re.sub(r"_p\d+$", "", n)
    return d.replace("SCS2_", "SCS_")          # SCS / SCS2 are two pages of the same system


groups = {}
for n in names: groups.setdefault(doc(n), []).append(n)
order = sorted(groups, key=lambda g: (-sum(nsym[n] for n in groups[g]), g))
A, B = [], []
for g in order:                                   # greedy: put into the half with fewer symbols
    sa, sb = sum(nsym[n] for n in A), sum(nsym[n] for n in B)
    (A if sa <= sb else B).extend(groups[g])
json.dump({"A_train": sorted(A), "B_test": sorted(B)}, open(os.path.join(os.path.dirname(__file__), "split.json"), "w"), indent=1, ensure_ascii=False)
print(f"document groups {len(groups)} | A(train) {len(A)} pages {sum(nsym[n] for n in A)} symbols | B(test) {len(B)} pages {sum(nsym[n] for n in B)} symbols")
for tag, S in (("A", A), ("B", B)): print(tag, [n.replace("pseudo_", "")[:28] for n in S])
