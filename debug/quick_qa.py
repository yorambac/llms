"""
Quick 10-question sanity check for the instruction-tuned model.
Usage: python debug/quick_qa.py [--ckpt path]
"""

import argparse, sys, torch, torch.nn as nn, torch.nn.functional as F, tiktoken

sys.path.insert(0, ".")
from train_250m import GPT

QUESTIONS = [
    ("Name the capital of France.", ""),
    ("Calculate the product of 12 and 8.", ""),
    ("List three planets in our solar system.", ""),
    ("Explain what H2O stands for.", ""),
    ("Write a one-sentence definition of gravity.", ""),
    ("State the boiling point of water in Celsius.", ""),
    ("Name the author of Romeo and Juliet.", ""),
    ("Give the opposite of the word 'hot'.", ""),
    ("Complete the following sentence.", "The sun rises in the..."),
    ("Translate the sentence to French.", "Hello, how are you?"),
]

def build_prompt(instruction, inp):
    inp = inp.strip()
    if inp and inp.lower() != "<noinput>":
        return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"

def generate(model, enc, prompt, max_tokens=80, temperature=0.7, top_k=50, device="cuda"):
    ids = enc.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_tokens):
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(idx[:, -1024:])
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            if nxt.item() == enc.eot_token:
                break
            idx = torch.cat([idx, nxt], dim=1)
    return enc.decode(idx[0, len(ids):].tolist()).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/important/sft_alpaca_best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading {args.ckpt} on {args.device}...", flush=True)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model = GPT().to(args.device)
    model.load_state_dict(sd)
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    print(f"Loaded. val_loss={ckpt.get('val_loss', '?')}\n")
    print("=" * 60)

    for i, (instruction, inp) in enumerate(QUESTIONS, 1):
        prompt = build_prompt(instruction, inp)
        response = generate(model, enc, prompt, device=args.device)
        label = f"{instruction}" + (f" [{inp}]" if inp else "")
        print(f"Q{i}: {label}")
        print(f"A:  {response}")
        print()

if __name__ == "__main__":
    main()
