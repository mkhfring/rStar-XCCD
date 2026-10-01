"""Paired bootstrap: SCB (python-only, tested rule) vs single-pass baselines on the same CLCCD pairs."""
import json,sys,glob,random
sys.path.insert(0,'.')
from rescore_runs import predict_tested
from evaluate_clone_results import normalize_label
def f1(g,p):
    tp=sum(a=='clone' and b=='clone' for a,b in zip(g,p)); fp=sum(a!='clone' and b=='clone' for a,b in zip(g,p)); fn=sum(a=='clone' and b!='clone' for a,b in zip(g,p))
    return 2*tp/(2*tp+fp+fn) if tp else 0.0
random.seed(0)
for L in ['java','rust']:
    scb={}
    for l in open(glob.glob(f'eval_data/test_python_{L}_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.ablation-pyonly.*.jsonl')[0]):
        r=json.loads(l)
        if 'rstar' in r: scb[r['index']]=(r['answer'],predict_tested(r['rstar']))
    for base in ['qwen3-8b','qwen3-4b']:
        bp={}
        for l in open(f'pure_inference/offline_results/test_python_{L}_CLCCD.jsonl_{base}_inference_result.jsonl'):
            r=json.loads(l)
            if 'rstar' not in r: continue
            fa=r['rstar']['1']['final_answer']; bp[r['idx']]=(r['answer'],normalize_label(fa) or 'none')
        keys=[k for k in scb if k in bp]
        assert all(scb[k][0]==bp[k][0] for k in keys)
        g=[scb[k][0] for k in keys]; a=[scb[k][1] for k in keys]; b=[bp[k][1] for k in keys]
        d0=f1(g,a)-f1(g,b); n=len(keys); ds=[]
        for _ in range(10000):
            ix=[random.randrange(n) for _ in range(n)]
            G=[g[i] for i in ix]; ds.append(f1(G,[a[i] for i in ix])-f1(G,[b[i] for i in ix]))
        ds.sort(); p=sum(x<=0 for x in ds)/len(ds)
        print(f"{L} SCB {f1(g,a):.4f} vs {base} {f1(g,b):.4f} (matched {n}) diff {d0:+.4f} 95% CI [{ds[250]:+.4f}, {ds[9750]:+.4f}] P(diff<=0)={p:.4f}")
