"""50 UNSEEN-tool rows, real descriptions vs off-domain random ones.

Same rows, same payloads, same gold. The ONLY change is the Danish prose in
the catalogue's `parameters`/`returns` descriptions, replaced with text about
postcodes, wind speed and VAT -- nothing to do with the tool.

If descriptions are a field->meaning map, this should wreck grounding.
If they are only topical context, the score should barely move while the
ANSWERS visibly change register.
"""
import itertools, json, os, sys, time, torch
sys.path.insert(0,"scripts")
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from train_sft_packed import format_conversation
from esperanto_lm.downstream_eval_callback import DownstreamEvaluator as E
CK=sys.argv[1]; N=int(os.environ.get("N","50")); BS=int(os.environ.get("BS","10"))
SEP="Værktøjer:"; dec=json.JSONDecoder()
JUNK=["Postnummeret for afsenderens folkeregisteradresse",
      "Antal sider i den trykte udgave af bogen",
      "Vindstyrke målt i meter per sekund ved kysten",
      "Navnet på den ansvarlige sagsbehandler",
      "Momssats anvendt ved beregning af fakturaen",
      "Farvekoden på emballagen i hexadecimal",
      "Afstanden til nærmeste jernbanestation i meter",
      "Serienummeret på den installerede måler"]
def randomise(cat):
    out=json.loads(json.dumps(cat)); pool=itertools.cycle(JUNK)
    for t in out:
        for blk in ("parameters","returns"):
            props=(t.get(blk) or {}).get("properties") or {}
            for k in props:
                if isinstance(props[k],dict): props[k]["description"]=next(pool)
        if isinstance(t.get("description"),str): t["description"]=next(pool)
    return out
ds=load_dataset("jensjepsen/danish-tool-dialogues-v9","sft",split="eval_unseen_tools")
items=[]
for r in ds:
    m=r["messages"]; h=m[0]["content"]
    if SEP not in h: continue
    ri=next((i for i,x in enumerate(m) if x["role"]=="tool_result"), None)
    if ri is None or ri+1>=len(m) or m[ri+1]["role"]!="assistant": continue
    gold=(m[ri+1].get("content") or "").strip()
    if not gold: continue
    body=h.split(SEP,1)[1].lstrip()
    try: cat,end=dec.raw_decode(body)
    except Exception: continue
    q=body[end:].strip()
    real=[dict(x) for x in m[:ri+1]]
    rand=[dict(x) for x in m[:ri+1]]
    rand[0]=dict(rand[0], content=f"{SEP}\n{json.dumps(randomise(cat),ensure_ascii=False)}\n\n{q}")
    items.append({"real":format_conversation(real)+" <|assistant|>",
                  "rand":format_conversation(rand)+" <|assistant|>",
                  "result":m[ri]["content"],"gold":gold})
    if len(items)>=N: break
print(f"{len(items)} unseen rows", flush=True)
tok=AutoTokenizer.from_pretrained(CK)
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(CK,torch_dtype=torch.float16).cuda().eval()
eos=[x for x in (tok.eos_token_id,tok.convert_tokens_to_ids("<|end|>")) if x is not None]
def run(key):
    outs=[]; t0=time.time(); prev=tok.padding_side; tok.padding_side="left"
    for i in range(0,len(items),BS):
        ch=[it[key] for it in items[i:i+BS]]
        enc=tok(ch,return_tensors="pt",padding=True,truncation=True,max_length=4096,
                add_special_tokens=False,return_token_type_ids=False).to("cuda")
        with torch.no_grad():
            o=model.generate(**enc,max_new_tokens=200,do_sample=False,eos_token_id=eos,
                             pad_token_id=tok.pad_token_id,repetition_penalty=1.1)
        for j in range(len(ch)):
            outs.append(tok.decode(o[j][enc["input_ids"].shape[1]:],skip_special_tokens=True).strip())
        print(f"  [{key}] {min(i+BS,len(items))}/{len(items)} {time.time()-t0:.0f}s",flush=True)
    tok.padding_side=prev; return outs
OUT="scratch/eyeball/randomdesc_unseen_50.txt"
open(OUT,"w").close()
res={}
for key in ("real","rand"):
    preds=run(key); sc=[]
    for it,pr in zip(items,preds):
        it[f"pred_{key}"]=pr
        vals=E._result_values(it["result"])
        if not vals: continue
        if E._is_echo(pr,it["result"]): sc.append(0.0)
        else:
            g={v for v in vals if E._mentions(it["gold"],v)}
            p={v for v in vals if E._mentions(pr,v)}
            sc.append(E._pair_f1(p,g))
        it[f"score_{key}"]=sc[-1]
        # STREAM. The previous version wrote the file only at the end and
        # printed one score per arm, so a 20-minute run showed nothing until
        # it was over. Every scored row now lands immediately.
        if key=="rand":
            with open(OUT,"a") as fh:
                fh.write("="*90+f"\nPAYLOAD: {it['result'][:200]}\nGULD   : {it['gold'][:160]}\n"
                         f"REAL  ({it.get('score_real',0):.2f}): {it.get('pred_real','')[:160]}\n"
                         f"RANDOM({it.get('score_rand',0):.2f}): {it.get('pred_rand','')[:160]}\n")
        if len(sc) % 10 == 0:
            print(f"    [{key}] running mean over {len(sc)}: {100*sum(sc)/len(sc):.1f}%", flush=True)
    res[key]=sum(sc)/len(sc)
    print(f"  => {key}: grounded {100*res[key]:.1f}% (n={len(sc)})",flush=True)
changed=sum(1 for it in items if it.get("pred_real","")!=it.get("pred_rand",""))
print(f"\n  REAL descriptions   {100*res['real']:.1f}%")
print(f"  RANDOM descriptions {100*res['rand']:.1f}%")
print(f"  delta               {100*(res['rand']-res['real']):+.1f} pp")
print(f"  answers that CHANGED at all: {changed}/{len(items)}")
print(f"-> {OUT}", flush=True)
