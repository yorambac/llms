"""
Stream FineWeb sample-10BT from HuggingFace, tokenise with GPT-2 tokenizer,
and save train/val splits as flat uint16 binary files.

Target: ~220M tokens total (200M train, 20M val).
Usage: python prepare_data.py
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from pathlib import Path
import time
from tqdm import tqdm

TRAIN_TOKENS = 200_000_000
VAL_TOKENS   =  20_000_000
TOTAL_TARGET = TRAIN_TOKENS + VAL_TOKENS
DATA_DIR     = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TRAIN_FILE = DATA_DIR / "train.bin"
VAL_FILE   = DATA_DIR / "val.bin"

def main():
    if TRAIN_FILE.exists() and VAL_FILE.exists():
        train_sz = TRAIN_FILE.stat().st_size // 2
        val_sz   = VAL_FILE.stat().st_size // 2
        print(f"Data already prepared: train={train_sz:,} tokens, val={val_sz:,} tokens")
        return

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    print(f"Streaming FineWeb sample-10BT (target {TOTAL_TARGET/1e6:.0f}M tokens)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    train_buf, val_buf = [], []
    train_count = val_count = 0
    t0 = time.time()

    with tqdm(total=TOTAL_TARGET, unit="tok", unit_scale=True,
              desc="Tokenising FineWeb", dynamic_ncols=True) as pbar:
        for doc in ds:
            if train_count + val_count >= TOTAL_TARGET:
                break

            ids = enc.encode_ordinary(doc["text"])
            ids.append(eot)

            if val_count < VAL_TOKENS:
                val_buf.extend(ids)
                val_count += len(ids)
            else:
                train_buf.extend(ids)
                train_count += len(ids)

            pbar.update(len(ids))
            elapsed = time.time() - t0
            if elapsed > 0:
                pbar.set_postfix({"tok/s": f"{(train_count+val_count)/elapsed/1e6:.2f}M"})

    print(f"Saving {train_count/1e6:.1f}M train tokens -> {TRAIN_FILE}")
    np.array(train_buf[:TRAIN_TOKENS], dtype=np.uint16).tofile(TRAIN_FILE)

    print(f"Saving {val_count/1e6:.1f}M val tokens -> {VAL_FILE}")
    np.array(val_buf[:VAL_TOKENS], dtype=np.uint16).tofile(VAL_FILE)

    print(f"Done in {time.time()-t0:.0f}s")
    os._exit(0)  # avoid HF datasets GIL crash on finalization


if __name__ == "__main__":
    main()
