"""Paired bootstrap: SCB (Qwen3-4B, python-only, tested rule) vs single-call CLCCD-paper
prompts (eval_data/paper_prompt_replication) on the same CLCCD pairs, aligned by index.
Unanswered baseline pairs count as wrong (a clone becomes FN; a non-clone stays negative)."""
import json,sys,glob,random
sys.path.insert(0,'.')
from rescore_runs import predict_tested
def f1(g,p):
    tp=sum(a=='clone' and b=='clone' for a,b in zip(g,p)); fp=sum(a!='clone' and b=='clone' for a,b in zip(g,p)); fn=sum(a=='clone' and b!='clone' for a,b in zip(g,p))
    return 2*tp/(2*tp+fp+fn) if tp else 0.0
random.seed(0)
BEST={('java','Qwen3-8B'):'sp2',('java','Qwen3-4B'):'sp2',('java','Qwen2.5-Coder-7B-Instruct'):'sp2',
      ('rust','Qwen3-8B'):'sp2',('rust','Qwen3-4B'):'sp2',('rust','Qwen2.5-Coder-7B-Instruct'):'sp2'}  # all chosen on the E1 dev set (2026-09-29)
for (L,m),p in BEST.items():
    scb={}
    for l in open(glob.glob(f'eval_data/test_python_{L}_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.ablation-pyonly.*.jsonl')[0]):
        r=json.loads(l)
        if 'rstar' in r: scb[int(r['index'])]=(r['answer'],predict_tested(r['rstar']))
    bp={}
    for l in open(sorted(glob.glob(f'eval_data/paper_prompt_replication/test_python_{L}_CLCCD.jsonl.{m}.{p}.2026*.jsonl'))[0]):
        r=json.loads(l); bp[int(r['index'])]=(r['answer'].lower(),r.get('predicted') or 'none')
    keys=sorted(scb); assert set(keys)==set(bp) and all(scb[k][0]==bp[k][0] for k in keys)
    g=[scb[k][0] for k in keys]; a=[scb[k][1] for k in keys]; b=[bp[k][1] for k in keys]
    n=len(keys); ds=[]
    for _ in range(10000):
        ix=[random.randrange(n) for _ in range(n)]; G=[g[i] for i in ix]
        ds.append(f1(G,[a[i] for i in ix])-f1(G,[b[i] for i in ix]))
    ds.sort()
    print(f"{L} SCB {f1(g,a):.4f} vs {m}/{p} {f1(g,b):.4f} (n={n}) diff {f1(g,a)-f1(g,b):+.4f} 95% CI [{ds[250]:+.4f}, {ds[9750]:+.4f}]")
