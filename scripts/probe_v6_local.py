"""Quick varied probes on the local v6 checkpoint. fp16 on Pascal."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CKPT = "/home/jepsen/src/espllm/runs/sft/da_v6_mix9wpreword/final"

tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float16).cuda().eval()
end_id = tok.convert_tokens_to_ids("<|end|>")
stop = [tok.eos_token_id, end_id]

prompts = [
    ("math-natural (v5 failed this)",
     "En butik sælger et par sko til 800 kr. med 20% rabat. Hvad er den oprindelige pris?"),
    ("math-natural rate",
     "En cyklist kører 45 km på 3 timer. Hvor lang tid tager det at køre 120 km ved samme fart?"),
    ("simple arith",
     "Hvad er 17 * 24?"),
    ("gsm8k-style age",
     "Katrine er dobbelt så gammel som Simon. Tilsammen er de 39 år. Hvor gammel er den yngste?"),
    ("wiki fact",
     "Hvem malede Mona Lisa?"),
    ("civics",
     "Hvad hedder Danmarks statsminister?"),
    ("IF: ends with phrase + no commas",
     "Skriv en kort tekst om vintersport. Ingen kommaer i svaret. Slut med sætningen 'Tak for din tid'."),
    ("IF: numbered list + all-caps",
     "Skriv præcis 3 punkter i en nummereret liste om Danmarks flag. Hele svaret skal være med STORE BOGSTAVER."),
    ("open-ended writing",
     "Skriv en kort besked til min ven om vejret i dag."),
    ("summarization",
     "Kort resumé af følgende tekst: København er Danmarks hovedstad og største by. Byen ligger på øerne Sjælland og Amager. Den blev grundlagt i det 12. århundrede."),
]

for label, p in prompts:
    prompt = f"<|user|> {p} <|assistant|>"
    ids = tok(prompt, return_tensors="pt", return_token_type_ids=False).to("cuda")
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=200, do_sample=False,
                             pad_token_id=tok.eos_token_id, eos_token_id=stop)
    gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).replace("<|end|>", "").strip()
    print("=" * 72)
    print(f"[{label}]")
    print("Q:", p)
    print("A:", gen)
    print()
