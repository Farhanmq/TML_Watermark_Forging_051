#!/usr/bin/env python3
"""
Sweep v4: StegaStamp + StableSignature via correct LFS URLs, plus HiDDeN via MBRS/HuggingFace.
Target: identify codec for WM_3, WM_4, WM_5, WM_6.
"""
import sys, os, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

CATS = [("WM_1",1,25),("WM_2",26,50),("WM_3",51,75),("WM_4",76,100),
        ("WM_5",101,125),("WM_6",126,150),("WM_7",151,175),("WM_8",176,200)]
AGREE_THRESH = 0.80
root = Path("watermarked_sources")
ckpt_dir = Path("/tmp/waves_ckpts"); ckpt_dir.mkdir(exist_ok=True)

# Correct LFS URLs (media.githubusercontent.com)
STEGA_URL  = "https://media.githubusercontent.com/media/umd-huang-lab/WAVES/main/decoders/stega_stamp.onnx"
STABLE_URL = "https://media.githubusercontent.com/media/umd-huang-lab/WAVES/main/decoders/stable_signature.onnx"

def download(url, dest, min_size=1_000_000):
    if dest.exists() and dest.stat().st_size > min_size:
        print(f"  {dest.name} already cached ({dest.stat().st_size//1024//1024} MB)")
        return dest
    print(f"  Downloading {dest.name} from LFS...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Done ({dest.stat().st_size//1024//1024} MB)")
    return dest

def load_rgb(p): return Image.open(p).convert("RGB")

def sweep(name, decode_fn, nbits):
    print(f"\n--- {name} ({nbits} bits) ---")
    for wm, start, stop in CATS:
        paths = sorted((root/wm).glob("*.png"))
        msgs = []
        for p in paths:
            try:
                bits = decode_fn(load_rgb(p))
                if bits is not None:
                    msgs.append(np.array(bits, dtype=int).flatten()[:nbits])
            except Exception:
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
    print("\n=== StegaStamp (100 bits, 400x400) ===")
    try:
        import onnxruntime as ort
        ckpt = download(STEGA_URL, ckpt_dir/"stega_stamp.onnx")
        sess = ort.InferenceSession(str(ckpt), providers=["CPUExecutionProvider"])
        print(f"  Loaded OK, inputs: {[i.name for i in sess.get_inputs()]}")

        def decode(img):
            img = ImageOps.fit(img, (400, 400))
            inp = np.array(img, dtype=np.float32)[None] / 255.0
            out = sess.run(None, {"image": inp, "secret": np.zeros((1,100), dtype=np.float32)})
            return out[2].flatten().astype(bool).astype(int)

        sweep("StegaStamp", decode, nbits=100)
    except Exception as e:
        print(f"StegaStamp failed: {e}")

# ── Stable Signature ──────────────────────────────────────────────────────────
def run_stable_sig():
    print("\n=== StableSignature (48 bits, 256x256) ===")
    try:
        import onnxruntime as ort
        ckpt = download(STABLE_URL, ckpt_dir/"stable_signature.onnx")
        sess = ort.InferenceSession(str(ckpt), providers=["CPUExecutionProvider"])
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def decode(img):
            img = img.resize((256, 256), Image.LANCZOS)
            arr = (np.array(img, dtype=np.float32)/255.0 - mean) / std
            inp = arr.transpose(2,0,1)[None]
            out = sess.run(None, {"image": inp})
            return (out[0].flatten() > 0).astype(int)

        sweep("StableSignature", decode, nbits=48)
    except Exception as e:
        print(f"StableSignature failed: {e}")

# ── HiDDeN via MBRS HuggingFace checkpoint ───────────────────────────────────
def run_hidden():
    print("\n=== HiDDeN-style decoder (via MBRS HuggingFace) ===")
    try:
        from huggingface_hub import hf_hub_download
        import torch
        import torch.nn as nn

        # Try loading a HiDDeN decoder from known HuggingFace repos
        # adobe-research/trustmark uses a similar backbone; try MBRS
        # The standard HiDDeN decoder: Conv blocks, output 30/48/64 bits
        print("  HiDDeN: no public pretrained checkpoint found for blind decode — skipping")
        print("  (HiDDeN requires knowing the encoder key/config)")
    except Exception as e:
        print(f"HiDDeN failed: {e}")

# ── RivaGAN with more bit lengths ─────────────────────────────────────────────
def run_rivagan_extended():
    print("\n=== RivaGAN extended bit lengths ===")
    try:
        import cv2
        from imwatermark import WatermarkDecoder
        def load_bgr(p): return cv2.imread(str(p))

        # RivaGAN is fixed 32 bits, but try anyway on all folders
        dec = WatermarkDecoder("bits", 32)
        for wm, start, stop in CATS:
            paths = sorted((root/wm).glob("*.png"))
            rows = []
            for p in paths:
                try:
                    rows.append(np.array(dec.decode(load_bgr(p), "rivaGan"), dtype=int))
                except Exception:
                    pass
            if not rows: continue
            st = np.stack(rows)
            maj = (st.mean(0) >= 0.5).astype(int)
            agree = float((st == maj).mean())
            flag = "  <-- MATCH" if agree >= AGREE_THRESH else ""
            print(f"  {wm}: agree={agree:.3f}{flag}  msg={''.join(map(str,maj.tolist()))}")
    except Exception as e:
        print(f"RivaGAN extended failed: {e}")

# ── Spatial frequency analysis of unknown folders ─────────────────────────────
def analyze_frequency():
    """Check if unknowns have Fourier-ring pattern (Tree-Ring / Gaussian Shading)."""
    print("\n=== Fourier frequency analysis ===")
    import cv2
    for wm in ["WM_3","WM_4","WM_5","WM_6"]:
        paths = sorted((root/wm).glob("*.png"))
        fmags = []
        for p in paths:
            img = np.array(Image.open(p).convert("L"), dtype=np.float32)
            f = np.fft.fftshift(np.fft.fft2(img))
            fmags.append(np.log1p(np.abs(f)))
        stack = np.stack(fmags)
        # Split-half NCC on Fourier magnitude
        A = stack[0::2].mean(0); B = stack[1::2].mean(0)
        a = A.flatten(); b = B.flatten()
        ncc = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)
        print(f"  {wm}: Fourier-mag NCC={ncc:.4f}")

if __name__ == "__main__":
    run_stegastamp()
    run_stable_sig()
    run_rivagan_extended()
    analyze_frequency()
    print("\n=== done ===")
