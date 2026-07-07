#!/usr/bin/env python3
"""
Targeted sweep: confirm TrustMark C/P/B assignments for each unknown folder.
Skips ECC (use_ECC=False) so all images decode regardless of detect confidence.
Also checks WM_4, WM_8 which had no valid decodes before.
"""
import sys, numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter

CATS_UNK = [("WM_3",51,75),("WM_4",76,100),("WM_5",101,125),("WM_6",126,150),("WM_8",176,200)]
CATS_ALL = [("WM_1",1,25),("WM_2",26,50),("WM_3",51,75),("WM_4",76,100),
            ("WM_5",101,125),("WM_6",126,150),("WM_7",151,175),("WM_8",176,200)]
root = Path("watermarked_sources")
AGREE_THRESH = 0.80

def load_rgb(p): return Image.open(p).convert("RGB")

def sweep_trustmark(variant, cats=CATS_ALL, use_ecc=False):
    from trustmark import TrustMark
    print(f"\n=== TrustMark-{variant} (use_ECC={use_ecc}) ===")
    try:
        tm = TrustMark(use_ECC=use_ecc, verbose=False, model_type=variant)
    except Exception as e:
        print(f"  Load failed: {e}")
        return

    for wm, start, stop in cats:
        paths = sorted((root/wm).glob("*.png"))
        msgs = []
        for p in paths:
            try:
                result = tm.decode(load_rgb(p), MODE="binary")
                secret, present, _ = result
                if secret is not None:
                    msgs.append(np.array([int(b) for b in secret], dtype=int))
            except Exception:
                pass
        if not msgs:
            print(f"  {wm}: no valid decodes")
            continue
        # Handle variable-length outputs
        lengths = [len(m) for m in msgs]
        most_common_len = Counter(lengths).most_common(1)[0][0]
        msgs = [m for m in msgs if len(m) == most_common_len]
        if not msgs:
            print(f"  {wm}: inconsistent output lengths")
            continue
        st = np.stack(msgs)
        maj = (st.mean(0) >= 0.5).astype(int)
        agree = float((st == maj).mean())
        flag = "  <-- MATCH" if agree >= AGREE_THRESH else ""
        n = len(msgs)
        print(f"  {wm}: agree={agree:.3f}  n={n}/{len(paths)}  len={most_common_len}{flag}  msg={''.join(map(str,maj.tolist()))[:48]}")

if __name__ == "__main__":
    # Run all 4 TrustMark variants with ECC disabled (forces decode on every image)
    for v in ["Q", "B", "C", "P"]:
        sweep_trustmark(v, cats=CATS_ALL, use_ecc=False)
    print("\n=== done ===")
