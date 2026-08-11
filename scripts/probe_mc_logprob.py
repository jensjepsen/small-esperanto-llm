"""Run sciq/citmc/arc MC-logprob eval on a checkpoint (same logic as callback)."""
import argparse, sys, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--n-sciq", type=int, default=1000)
    ap.add_argument("--n-citmc", type=int, default=720)
    ap.add_argument("--n-arc", type=int, default=1167)
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from esperanto_lm.mc_logprob_callback import MCLogprobCallback

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[args.dtype]
    print(f"loading {args.ckpt}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.ckpt)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=dtype).cuda().eval()

    cb = MCLogprobCallback(tokenizer=tok, n_sciq=args.n_sciq,
                            n_citmc=args.n_citmc, n_arc=args.n_arc)
    cb._load_sciq(); cb._load_citmc(); cb._load_arc()
    print(f"loaded  sciq={len(cb._sciq)}  citmc={len(cb._citmc)}  arc={len(cb._arc)}", flush=True)

    sciq_acc = cb._score_items(model, cb._sciq)
    print(f"  sciq_mc_logprob = {100*sciq_acc:.2f}%", flush=True)
    cit_acc = cb._score_items(model, cb._citmc)
    print(f"  citmc_logprob   = {100*cit_acc:.2f}%", flush=True)
    arc_acc = cb._score_items(model, cb._arc)
    print(f"  arc_logprob     = {100*arc_acc:.2f}%", flush=True)


if __name__ == "__main__":
    main()
