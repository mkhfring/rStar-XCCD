"""Rust CLCCD clone recall split by crate need (paper RQ4 CLCCD paragraph). Run from rstar-xccd/."""
import json,re,sys,glob
sys.path.insert(0,'.')
from rescore_runs import RULES
STD={'std','core','alloc','self','super','crate','Node','io'}
def needs(q):
    m=re.search(r"```rust\n(.*?)```",q,re.S); code=m.group(1) if m else ''
    roots=set(re.findall(r"^\s*(?:pub\s+)?use\s+:?:?([A-Za-z_]\w*)",code,re.M))|set(re.findall(r"extern\s+crate\s+(\w+)",code))
    return bool(roots-STD)
def run(pat,rule):
    f=[g for g in glob.glob(pat) if not g.endswith('_result')][0]
    rows=[r for r in map(json.loads,open(f)) if 'rstar' in r and r['answer']=='clone']
    out={}
    for k in (False,True):
        s=[r for r in rows if needs(r['question'])==k]
        hit=sum(RULES[rule](r['rstar'])=='clone' for r in s)
        out['crate' if k else 'build']=(hit,len(s),round(hit/max(1,len(s)),3))
    print(f.split('Qwen3-4B.')[1][:30],rule,out)
B='eval_data/test_python_rust_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.'
run(B+'clccd-ext.def*','current'); run(B+'clccd-ext-it8v3.*','current'); run(B+'ablation-pyonly.def*','tested')
