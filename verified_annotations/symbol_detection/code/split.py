"""把 65 张核验页按"文档"分成两半 (同一份文档的多页放同一半, 避免近似页泄漏), 两半符号数尽量相等.
A = 训练半, B = 测试半.  输出 split.json"""
import glob, json, os, re
R = "/home/suxinqi666/code/PnIDAgent_project"
LAB = f"{R}/line_results/labels_verified"
names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{LAB}/*.txt"))
nsym = {n: sum(1 for l in open(f"{LAB}/{n}.txt") if len(l.split()) == 5) for n in names}


def doc(n):
    d = re.sub(r"_p\d+$", "", n)
    return d.replace("SCS2_", "SCS_")          # SCS / SCS2 是同一系统的两页


groups = {}
for n in names: groups.setdefault(doc(n), []).append(n)
order = sorted(groups, key=lambda g: (-sum(nsym[n] for n in groups[g]), g))
A, B = [], []
for g in order:                                   # 贪心: 放进符号数少的那一半
    sa, sb = sum(nsym[n] for n in A), sum(nsym[n] for n in B)
    (A if sa <= sb else B).extend(groups[g])
json.dump({"A_train": sorted(A), "B_test": sorted(B)}, open(os.path.join(os.path.dirname(__file__), "split.json"), "w"), indent=1, ensure_ascii=False)
print(f"文档组 {len(groups)} | A(训练) {len(A)} 页 {sum(nsym[n] for n in A)} 符号 | B(测试) {len(B)} 页 {sum(nsym[n] for n in B)} 符号")
for tag, S in (("A", A), ("B", B)): print(tag, [n.replace("pseudo_", "")[:28] for n in S])
