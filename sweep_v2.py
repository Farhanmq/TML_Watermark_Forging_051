#!/usr/bin/env python3
"""
Broader codec sweep: StegaStamp + StableSignature ONNX (CPU-forced) + TrustMark B/C/Q variants
Agreement >= 0.80 across 25 source images = codec identified.
"""

import sys, os, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from collections import Counter

CATS = [("WM_1",1,25),("WM_2",26,50),("WM_3",51,75),("WM_4",76,100),
        ("WM_5",101,125),("WM_6",126,150),("WM_7",151,175),("WM_8",176,200)]
AGREE_THRESH = 0.80
root = Path("watermarked_sources")
ckpt_dir = Path("/tmp/waves_ckpts"); ckpt_dir.mkdir(exist_ok=True)

WAVES_BASE = "https://raw.githubusercontent.com/umd-huang-lab/WAVES/main/decoders"

def download(url, dest):
    if not dest.exists():
        print(f"  Downloading {dest.name}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Done ({dest.stat().st_size//1024} KB)")
    return dest

def load_rgb(p): return Image.open(p).convert("RGB")

def sweep_bits(name, decode_fn, nbits):
    print(f"\n--- {name} ({nbits} bits) ---")
    for wm, start, stop in CATS:
        paths = sorted((root/wm).glob("*.png"))
        msgs = []
        for p in paths:
            try:
                bits = decode_fn(load_rgb(p))
                if bits is not None:
                    msgs.append(np.array(bits, dtype=int).flatten()[:nbits])
            except Exception as e:
                pass
        if not msgs:
            print(f"  {wm}: no valid decodes")
            continue
        st = np.stack(msgs)
        maj = (st.mean(0) >= 0.5).astype(int)
        agree = float((st == maj).mean())
        flag = "  <-- MATCH" if agree >= AGREE_THRESH else ""
        print(f"  {wm}: agree={agree:.3f}{flag}  msg={''.join(map(str,maj.tolist()))[:48]}")

# ── StegaStamp ────────────────────────────────────────────────────────────────
def run_stegastamp():
    print("\n=== StegaStamp (100 bits, 400x400, CPU-only) ===")
    try:
        import onnxruntime as ort
        ckpt = download(f"{WAVES_BASE}/stega_stamp.onnx", ckpt_dir/"stega_stamp.onnx")
        so = ort.SessionOptions()
        sess = ort.InferenceSession(str(ckpt), sess_options=so,
                                    providers=["CPUExecutionProvider"])
        print(f"  Providers: {sess.get_providers()}")

        def decode(img):
            img = ImageOps.fit(img, (400, 400))
            inp = np.array(img, dtype=np.float32)[None] / 255.0  # (1,400,400,3)
            out = sess.run(None, {
                "image": inp,
                "secret": np.zeros((1, 100), dtype=np.float32),
            })
            return out[2].flatten().astype(bool).astype(int)

        sweep_bits("StegaStamp", decode, nbits=100)
    except Exception as e:
        print(f"StegaStamp failed: {e}")

# ── Stable Signature ──────────────────────────────────────────────────────────
def run_stable_sig():
    print("\n=== StableSignature (48 bits, 256x256, ImageNet-norm, CPU-only) ===")
    try:
        import onnxruntime as ort
        ckpt = download(f"{WAVES_BASE}/stable_signature.onnx", ckpt_dir/"stable_signature.onnx")
        so = ort.SessionOptions()
        sess = ort.InferenceSession(str(ckpt), sess_options=so,
                                    providers=["CPUExecutionProvider"])
        print(f"  Providers: {sess.get_providers()}")

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def decode(img):
            img = img.resize((256, 256), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - mean) / std
            inp = arr.transpose(2, 0, 1)[None]  # (1,3,256,256)
            out = sess.run(None, {"image": inp})
            return (out[0].flatten() > 0).astype(int)

        sweep_bits("StableSignature", decode, nbits=48)
    except Exception as e:
        print(f"StableSignature failed: {e}")

# ── TrustMark variants B and C ────────────────────────────────────────────────
def run_trustmark_variants():
    print("\n=== TrustMark variants (B, C, P) ===")
    try:
        from trustmark import TrustMark
        for variant in ["B", "C", "P"]:
            print(f"\n  TrustMark-{variant}:")
            try:
                tm = TrustMark(verbose=False, model_type=variant)
                def make_decode(tm_inst):
                    def decode(img):
                        try:
                            secret, present, _ = tm_inst.decode(img, MODE="binary")
                            if present:
                                return [int(b) for b in secret]
                        except:
                            pass
                        return None
                    return decode

                decode_fn = make_decode(tm)
                for wm, start, stop in CATS:
                    paths = sorted((root/wm).glob("*.png"))
                    msgs = []
                    for p in paths:
                        bits = decode_fn(load_rgb(p))
                        if bits is not None:
                            msgs.append(np.array(bits, dtype=int))
                    if not msgs:
                        print(f"    {wm}: no valid decodes")
                        continue
                    st = np.stack(msgs)
                    maj = (st.mean(0) >= 0.5).astype(int)
                    agree = float((st == maj).mean())
                    flag = "  <-- MATCH" if agree >= AGREE_THRESH else ""
                    print(f"    {wm}: agree={agree:.3f}{flag}  len={len(maj)}")
            except Exception as e:
                print(f"  TrustMark-{variant} failed: {e}")
    except ImportError as e:
        print(f"TrustMark not available: {e}")

# ── dwtDctSvd (already known to fail) ────────────────────────────────────────
def run_dwtdctsvd():
    print("\n=== dwtDctSvd (imwatermark, all bit lengths) ===")
    try:
        import cv2
        from imwatermark import WatermarkDecoder
        def load_bgr(p): return cv2.imread(str(p))

        for length in [16, 32, 48, 64, 96, 128]:
            print(f"\n  dwtDctSvd-{length}:")
            dec = WatermarkDecoder("bits", length)
            for wm, start, stop in CATS:
                paths = sorted((root/wm).glob("*.png"))
                rows = []
                for p in paths:
                    try:
                        rows.append(np.array(dec.decode(load_bgr(p), "dwtDctSvd"), dtype=int))
                    except Exception:
                        pass
                if not rows: continue
                st = np.stack(rows)
                maj = (st.mean(0) >= 0.5).astype(int)
                agree = float((st == maj).mean())
                flag = "  <-- MATCH" if agree >= AGREE_THRESH else ""
                if agree >= 0.75 or flag:
                    print(f"    {wm}: agree={agree:.3f}{flag}  msg={''.join(map(str,maj.tolist()))[:32]}")
                else:
                    print(f"    {wm}: agree={agree:.3f}")
    except Exception as e:
        print(f"dwtDctSvd failed: {e}")

# ── maxDct (imwatermark) ──────────────────────────────────────────────────────
def run_maxdct():
    print("\n=== maxDct sweep (imwatermark) ===")
    try:
        import cv2
        from imwatermark import WatermarkDecoder
        def load_bgr(p): return cv2.imread(str(p))

        for length in [8, 16, 32, 48, 64]:
            print(f"\n  maxDct-{length}:")
            dec = WatermarkDecoder("bits", length)
            for wm, start, stop in CATS:
                paths = sorted((root/wm).glob("*.png"))
                rows = []
                for p in paths:
                    try:
                        rows.append(np.array(dec.decode(load_bgr(p), "dwtDct"), dtype=int))
                    except Exception:
                        pass
                if not rows: continue
                st = np.stack(rows)
                maj = (st.mean(0) >= 0.5).astype(int)
                agree = float((st == maj).mean())
                flag = "  <-- MATCH" if agree >= AGREE_THRESH else ""
                if agree >= 0.80 or flag:
                    print(f"    {wm}: agree={agree:.3f}{flag}  msg={''.join(map(str,maj.tolist()))[:32]}")
                else:
                    print(f"    {wm}: agree={agree:.3f}")
    except Exception as e:
        print(f"maxDct failed: {e}")

# ── Per-bit confidence analysis for residual folders ─────────────────────────
def analyze_residual_quality():
    """Check how confident each bit is in the median residual per unknown folder."""
    print("\n=== Residual bit confidence analysis ===")
    import cv2

    # Use nlm5 denoising
    for wm in ["WM_3", "WM_4", "WM_5", "WM_6", "WM_8"]:
        paths = sorted((root/wm).glob("*.png"))
        residuals = []
        for p in paths:
            img = cv2.imread(str(p)).astype(np.float32)
            denoised = cv2.fastNlMeansDenoisingColored(
                np.clip(img,0,255).astype(np.uint8), None, 5, 5, 7, 21
            ).astype(np.float32)
            residuals.append(img - denoised)

        stack = np.stack(residuals)  # (25, H, W, 3)
        # Split-half test
        A = np.median(stack[0::2], axis=0)
        B = np.median(stack[1::2], axis=0)
        median_res = np.median(stack, axis=0)

        # Normalized cross-correlation
        a_flat = A.flatten(); b_flat = B.flatten()
        ncc = np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-9)
        rms = float(np.sqrt((median_res**2).mean()))

        # Check sign consistency per pixel (proxy for bit confidence)
        sign_consistent = float((np.sign(A) == np.sign(B)).mean())

        print(f"  {wm}: NCC={ncc:.3f}  rms={rms:.3f}  sign_consistent={sign_consistent:.3f}")

if __name__ == "__main__":
    run_stegastamp()
    run_stable_sig()
    run_trustmark_variants()
    run_maxdct()
    run_dwtdctsvd()
    analyze_residual_quality()
    print("\n=== done ===")
