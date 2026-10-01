import json,sys,glob,re,collections
sys.path.insert(0,'.')
from rescore_runs import predict_tested, tested, f1
from evaluate_clone_results import predict_label, node_sort_key, normalize_label
from trace_checks import chain, code_steps
runs={
 'test java':glob.glob('eval_data/test_python_java_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.ablation-pyonly.*.jsonl')[0],
 'test rust':glob.glob('eval_data/test_python_rust_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.ablation-pyonly.*.jsonl')[0],
 'dev java':glob.glob('eval_data/e16a/dev_python_java_codenet_depth_16.jsonl.mcts.Qwen3-4B.e1-dev-pyonly.*.jsonl')[0],
 'dev rust':glob.glob('eval_data/e16a/dev_python_rust_codenet_depth_16.jsonl.mcts.Qwen3-4B.e1-dev-pyonly.*.jsonl')[0],
}
for name,f in runs.items():
    rows=[json.loads(l) for l in open(f)]; rows=[r for r in rows if r.get('rstar')]
    why=collections.Counter(); proxy=[]; tst=[]; flipout=collections.Counter()
    for r in rows:
        t=r['rstar']; gold=r['answer']
        num={k:v for k,v in t.items() if node_sort_key(k) is not None}
        leaves=[]
        for tag,n in num.items():
            fa=(n.get('final_answer') or '').strip(); lab=normalize_label(fa) if fa else None
            if lab: leaves.append((tag,lab,tested(num,tag)))
        p=predict_tested(t); tst.append((gold,p))
        # proxy: did the model run ANY real test in any leaf path? (ignores what the test showed and what it concluded)
        anytest=any(tt for _,_,tt in leaves) or any(l=='clone' for _,l,_ in leaves)
        proxy.append((gold,'clone' if anytest else 'non-clone'))
        if not leaves: why['no leaf vote -> fallback',gold]+=1
        elif any(l=='clone' for _,l,_ in leaves): why['model itself said clone',gold]+=1
        elif any(tt for _,_,tt in leaves): why['FLIP: tested non-clone -> clone',gold]+=1
        else: why['all non-clone leaves untested (shortcut)',gold]+=1
    tot=len(rows)
    print(f"\n=== {name} (n={tot}) tested F1 {f1(tst)[0]:.3f} | proxy 'model ran any test' F1 {f1(proxy)[0]:.3f} | agreement {sum(a[1]==b[1] for a,b in zip(tst,proxy))/tot:.3f}")
    for k in sorted({k for k,_ in why}):
        print(f"   {k:42s} gold clone {why[k,'clone']:4d} | gold non-clone {why[k,'non-clone']:4d}")
