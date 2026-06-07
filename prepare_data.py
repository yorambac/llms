"""
Stream FineWeb sample-100BT from HuggingFace, tokenise with GPT-2 tokenizer,
and save train/val splits as flat uint16 binary files.

Target: ~10.52B tokens total (10.5B train, 20M val). Uses sample-100BT to avoid cycling.
Writes directly to memmap — constant RAM usage regardless of dataset size.
Supports resume: if partial files exist, continues from where it left off.

Usage: python prepare_data.py
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from pathlib import Path
import time
from tqdm import tqdm

TRAIN_TOKENS = 10_500_000_000
VAL_TOKENS   =    20_000_000
TOTAL_TARGET = TRAIN_TOKENS + VAL_TOKENS
DATA_DIR     = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TRAIN_FILE   = DATA_DIR / "train.bin"
VAL_FILE     = DATA_DIR / "val.bin"
PROGRESS_FILE = DATA_DIR / "prepare_progress.txt"


def read_progress():
    if PROGRESS_FILE.exists():
        parts = PROGRESS_FILE.read_text().strip().split()
        return int(parts[0]), int(parts[1])  # val_pos, train_pos
    return 0, 0


def write_progress(val_pos, train_pos):
    PROGRESS_FILE.write_text(f"{val_pos} {train_pos}\n")


def main():
    val_pos, train_pos = read_progress()
    if train_pos >= TRAIN_TOKENS and val_pos >= VAL_TOKENS:
        print(f"Data already prepared: train={train_pos:,} tokens, val={val_pos:,} tokens")
        return

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    resuming = val_pos > 0 or train_pos > 0
    if resuming:
        print(f"Resuming: val={val_pos:,} train={train_pos:,} tokens already written")
    else:
        print(f"Streaming FineWeb sample-100BT (target {TOTAL_TARGET/1e6:.0f}M tokens)...")

    # Pre-allocate memmap files (or open existing ones)
    val_mm   = np.memmap(VAL_FILE,   dtype=np.uint16, mode='r+' if VAL_FILE.exists()   else 'w+', shape=(VAL_TOKENS,))
    train_mm = np.memmap(TRAIN_FILE, dtype=np.uint16, mode='r+' if TRAIN_FILE.exists() else 'w+', shape=(TRAIN_TOKENS,))

    already_done = val_pos + train_pos
    t0 = time.time()

    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-100BT",
        split="train",
        streaming=True,
    )

    # Skip already-processed docs by skipping tokens (approximate via skip)
    # For resume: just fast-forward the iterator token count
    skipped = 0
    if resuming:
        print(f"Fast-forwarding through {already_done/1e9:.2f}B already-tokenised tokens...")

    last_print = t0

    import sys
    with tqdm(total=TOTAL_TARGET, unit="tok", unit_scale=True,
              desc="Tokenising FineWeb", dynamic_ncols=True,
              initial=already_done, file=sys.stderr) as pbar:
        for doc in ds:
            ids = enc.encode_ordinary(doc["text"])
            ids.append(eot)
            n = len(ids)

            # Fast-forward: skip docs already tokenised
            if skipped + n <= already_done:
                skipped += n
                continue
            # Partial overlap at resume boundary
            if skipped < already_done:
                ids = ids[already_done - skipped:]
                skipped = already_done

            arr = np.array(ids, dtype=np.uint16)

            # Fill val first
            if val_pos < VAL_TOKENS:
                take = min(len(arr), VAL_TOKENS - val_pos)
                val_mm[val_pos:val_pos + take] = arr[:take]
                val_pos += take
                arr = arr[take:]
                if val_pos == VAL_TOKENS:
                    val_mm.flush()

            # Then train
            if len(arr) > 0 and train_pos < TRAIN_TOKENS:
                take = min(len(arr), TRAIN_TOKENS - train_pos)
                train_mm[train_pos:train_pos + take] = arr[:take]
                train_pos += take

            pbar.update(n)
            elapsed = time.time() - t0

            # Print clean status line every 60s (readable in log when tqdm → /dev/null)
            now = time.time()
            if now - last_print >= 60:
                total_done = val_pos + train_pos
                pct = total_done / TOTAL_TARGET * 100
                rate = (total_done - already_done) / elapsed / 1e6 if elapsed > 0 else 0
                eta_s = (TOTAL_TARGET - total_done) / (rate * 1e6) if rate > 0 else 0
                print(f"[prepare] {pct:.1f}%  {total_done/1e9:.2f}B/{TOTAL_TARGET/1e9:.1f}B tok  "
                      f"{rate:.2f}M tok/s  ETA {eta_s/60:.0f}min", flush=True)
                last_print = now

            # Flush + save progress every 500M tokens
            if (val_pos + train_pos) % 500_000_000 < n:
                train_mm.flush()
                write_progress(val_pos, train_pos)

            if val_pos >= VAL_TOKENS and train_pos >= TRAIN_TOKENS:
                break

    train_mm.flush()
    val_mm.flush()
    write_progress(val_pos, train_pos)

    print(f"Done: {train_pos/1e9:.2f}B train tokens, {val_pos/1e6:.0f}M val tokens")
    print(f"Elapsed: {time.time()-t0:.0f}s")
    os._exit(0)


if __name__ == "__main__":
    main()
