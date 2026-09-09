"""Grounding on SEEN vs UNSEEN tools -- the split tool_answer never scored.

`_load_tool_answer` reads eval_seen_tools ONLY, justified by a comment saying
eval_unseen carries a follow-up answer on just 13% of rows (98 of 768). That
is stale: v8 has 466 answer-bearing unseen rows (61%) and v9 has 526 (68%).
So grounding on tools the model has never seen has never been measured, and
every tool_answer number we hold is a seen-tool number.

    uv run python scripts/ground_seen_vs_unseen.py <ckpt>          # N/BS via env
"""
import json, os, sys, time, torch
sys.path.insert(0,"scripts")
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from train_sft_packed import format_conversation
from esperanto_lm.downstream_eval_callback import DownstreamEvaluator as E
CK=sys.argv[1]; N=int(os.environ.get("N","220")); BS=int(os.environ.get("BS","16"))
tok=AutoTokenizer.from_pretrained(CK)
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(CK, torch_dtype=torch.float16).cuda().eval()
eos=[x for x in (tok.eos_token_id, tok.convert_tokens_to_ids("<|end|>")) if x is not None]
def items(split):
    ds=load_dataset("jensjepsen/danish-tool-dialogues-v9","sft",split=split)
    out=[]
    for r in ds:
        m=r["messages"]
        ri=next((i for i,x in enumerate(m) if x["role"]=="tool_result"), None)
        if ri is None or ri+1>=len(m) or m[ri+1]["role"]!="assistant": continue
        g=(m[ri+1].get("content") or "").strip()
        if not g: continue
        out.append((format_conversation(m[:ri+1])+" <|assistant|>", m[ri]["content"], g))
        if len(out)>=N: break
    return out
res={}
for split in ("eval_seen_tools","eval_unseen_tools"):
    its=items(split); preds=[]; t0=time.time()
    prev=tok.padding_side; tok.padding_side="left"
    for i in range(0,len(its),BS):
        ch=[x[0] for x in its[i:i+BS]]
        enc=tok(ch,return_tensors="pt",padding=True,truncation=True,max_length=4096,
                add_special_tokens=False,return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            o=model.generate(**enc,max_new_tokens=300,do_sample=False,eos_token_id=eos,
                             pad_token_id=tok.pad_token_id,repetition_penalty=1.1)
        for j in range(len(ch)):
            preds.append(tok.decode(o[j][enc["input_ids"].shape[1]:],skip_special_tokens=True).strip())
        print(f"  [{split}] {min(i+BS,len(its))}/{len(its)} {time.time()-t0:.0f}s",flush=True)
    tok.padding_side=prev
    sc=[]; ech=0; rows=[]
    for (p,resu,gold),pr in zip(its,preds):
        vals=E._result_values(resu)
        if not vals: continue
        if E._is_echo(pr,resu): sc.append(0.0); ech+=1
        else:
            g={v for v in vals if E._mentions(gold,v)}
            q={v for v in vals if E._mentions(pr,v)}
            sc.append(E._pair_f1(q,g))
        rows.append({"result":resu,"gold":gold,"pred":pr,"score":sc[-1]})
    res[split]=(sum(sc)/len(sc),len(sc),ech,rows)
    print(f"  => {split}: grounded {100*res[split][0]:.1f}% (n={len(sc)}, echo={ech})",flush=True)
import pathlib
pathlib.Path("scratch/eyeball").mkdir(parents=True,exist_ok=True)
for split,(m,n,e,rows) in res.items():
    with open(f"scratch/eyeball/ground_{split}.txt","w") as fh:
        fh.write(f"{split}  grounded {100*m:.1f}%  n={n}\n\n")
        for r in sorted(rows,key=lambda x:x["score"]):
            fh.write("="*90+f"\nscore {r['score']:.2f}\nPAYLOAD: {r['result'][:220]}\n"
                     f"GULD   : {r['gold'][:180]}\nMODEL  : {r['pred'][:180]}\n")
print(f"\n  SEEN   {100*res['eval_seen_tools'][0]:.1f}%")
print(f"  UNSEEN {100*res['eval_unseen_tools'][0]:.1f}%")
print(f"  delta  {100*(res['eval_unseen_tools'][0]-res['eval_seen_tools'][0]):+.1f} pp")
