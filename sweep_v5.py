#!/usr/bin/env python3
"""
sweep_v5: Fix StableSignature ONNX (inspect input names first), also try
RivaGAN on all folders, and do a detailed residual bit-confidence analysis
to understand the ceiling for WM_3/4/5/6.
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

STEGA_URL  = "https://media.githubusercontent.com/media/umd-huang-lab/WAVES/main/decoders/stega_stamp.onnx"
STABLE_URL = "https://media.githubusercontent.com/media/umd-huang-lab/WAVES/main/decoders/stable_signature.onnx"

def download(url, dest, min_size=100_000):
    if dest.exists() and dest.stat().st_size > min_size:
        print(f"  {dest.name} cached ({dest.stat().st_size//1024} KB)")
        return dest
    print(f"  Downloading {dest.name}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Done ({dest.stat().st_size//1024} KB)")
    return dest

def load_rgb(p): return Image.open(p).convert("RGB")

def sweep(name, decode_fn, nbits):
    print(f"\n--- {name} ({nbits} bits) ---")
    for wm, start, stop in CATS:
        paths = sorted((root/wm).glob("*.png"))
        msgs = []
        errors = []
        for p in paths:
            try:
                bits = decode_fn(load_rgb(p))
                if bits is not None:
                    msgs.append(np.array(bits, dtype=int).flatten()[:nbits])
            except Exception as e:
                errors.append(str(e))
        if not msgs:
            err_sample = errors[0] if errors else "unknown"
            print(f"  {wm}: no valid decodes (err: {err_sample[:80]})")
            continue
        st = np.stack(msgs)
        maj = (st.mean(0) >= 0.5).astype(int)
        agree = float((st == maj).mean())
        flag = "  <-- MATCH" if agree >= AGREE_THRESH else ""
        print(f"  {wm}: agree={agree:.3f}  n={len(msgs)}{flag}  msg={''.join(map(str,maj.tolist()))[:48]}")

# ── StableSignature: inspect model first ─────────────────────────────────────
def run_stable_sig():
    print("\n=== StableSignature (48 bits) ===")
    try:
        import onnxruntime as ort
        ckpt = download(STABLE_URL, ckpt_dir/"stable_signature.onnx")
        sess = ort.InferenceSession(str(ckpt), providers=["CPUExecutionProvider"])

        # Inspect the model's actual input/output names and shapes
        print("  Model inputs:")
        for inp in sess.get_inputs():
            print(f"    name={inp.name!r}  shape={inp.shape}  type={inp.type}")
        print("  Model outputs:")
        for out in sess.get_outputs():
            print(f"    name={out.name!r}  shape={out.shape}  type={out.type}")

        inp_name = sess.get_inputs()[0].name
        inp_shape = sess.get_inputs()[0].shape  # e.g. [1, 3, 256, 256] or [1, 256, 256, 3]

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Try both NCHW and NHWC based on shape
        def decode(img):
            # Determine expected spatial size from model shape
            shape = sess.get_inputs()[0].shape
            # shape could be [1, 3, H, W] or [1, H, W, 3]
            if len(shape) == 4:
                if shape[1] == 3:  # NCHW
                    H, W = shape[2], shape[3]
                    if H == -1 or H is None: H = 256
                    if W == -1 or W is None: W = 256
                    img_r = img.resize((int(W), int(H)), Image.LANCZOS)
                    arr = (np.array(img_r, dtype=np.float32)/255.0 - mean) / std
                    inp_arr = arr.transpose(2,0,1)[None]
                else:  # NHWC
                    H, W = shape[1], shape[2]
                    if H == -1 or H is None: H = 256
                    if W == -1 or W is None: W = 256
                    img_r = img.resize((int(W), int(H)), Image.LANCZOS)
                    arr = (np.array(img_r, dtype=np.float32)/255.0 - mean) / std
                    inp_arr = arr[None]
            else:
                img_r = img.resize((256, 256), Image.LANCZOS)
                arr = (np.array(img_r, dtype=np.float32)/255.0 - mean) / std
                inp_arr = arr.transpose(2,0,1)[None]

            out = sess.run(None, {inp_name: inp_arr})
            print(f"    output[0] shape={out[0].shape}  min={out[0].min():.3f}  max={out[0].max():.3f}  mean={out[0].mean():.3f}")
            return (out[0].flatten() > 0).astype(int)

        # Test one image first
        test_img = load_rgb(sorted((root/"WM_3").glob("*.png"))[0])
        print("\n  Test decode on WM_3/first image:")
        bits = decode(test_img)
        print(f"  bits (first 48): {''.join(map(str, bits[:48]))}")

        sweep("StableSignature", decode, nbits=48)
    except Exception as e:
        import traceback
        print(f"StableSignature failed: {e}")
        traceback.print_exc()

# ── RivaGAN on all folders ────────────────────────────────────────────────────
def run_rivagan():
    print("\n=== RivaGAN-32 on all folders ===")
    try:
        import cv2
        from imwatermark import WatermarkDecoder
        def load_bgr(p): return cv2.imread(str(p))
        dec = WatermarkDecoder("bits", 32)
        for wm, start, stop in CATS:
            paths = sorted((root/wm).glob("*.png"))
            rows = []
            for p in paths:
                try:
                    rows.append(np.array(dec.decode(load_bgr(p), "rivaGan"), dtype=int))
                except Exception:
                    pass
            if not rows:
                print(f"  {wm}: no decodes")
                continue
            st = np.stack(rows)
            maj = (st.mean(0) >= 0.5).astype(int)
            agree = float((st == maj).mean())
            flag = "  <-- MATCH" if agree >= AGREE_THRESH else ""
            print(f"  {wm}: agree={agree:.3f}{flag}  msg={''.join(map(str,maj.tolist()))}")
    except Exception as e:
        print(f"RivaGAN failed: {e}")

# ── Per-bit confidence for unknown folders ────────────────────────────────────
def analyze_residual_bits():
    """For each unknown folder, analyse how well each pixel is determined.
    A pixel that reliably differs from its denoised version across ALL 25 images
    is a high-confidence watermark pixel. This tells us the maximum achievable
    bit accuracy with residual injection."""
    print("\n=== Residual pixel confidence analysis ===")
    import cv2

    denoisers = {"WM_3": "nlm5", "WM_4": "median3", "WM_5": "wavelet", "WM_6": "bilat"}

    for wm, den_name in denoisers.items():
        paths = sorted((root/wm).glob("*.png"))
        residuals = []
        for p in paths:
            img = cv2.imread(str(p)).astype(np.float32)
            if den_name == "nlm5":
                smooth = cv2.fastNlMeansDenoisingColored(np.clip(img,0,255).astype(np.uint8),None,5,5,7,21).astype(np.float32)
            elif den_name == "median3":
                smooth = cv2.medianBlur(np.clip(img,0,255).astype(np.uint8),3).astype(np.float32)
            elif den_name == "bilat":
                smooth = cv2.bilateralFilter(np.clip(img,0,255).astype(np.uint8),5,50,50).astype(np.float32)
            elif den_name == "wavelet":
                from skimage.restoration import denoise_wavelet
                smooth = (denoise_wavelet(img/255., channel_axis=-1, rescale_sigma=True,
                                          method="BayesShrink", mode="soft")*255).astype(np.float32)
            residuals.append(img - smooth)

        stack = np.stack(residuals)   # (25, H, W, 3)
        median_res = np.median(stack, axis=0)
        A = np.median(stack[0::2], axis=0)
        B = np.median(stack[1::2], axis=0)

        # Sign consistency: fraction of pixels where A and B agree in sign
        sign_consist = float((np.sign(A) == np.sign(B)).mean())

        # NCC between A and B halves
        a_flat = A.flatten(); b_flat = B.flatten()
        ncc = float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat)*np.linalg.norm(b_flat)+1e-9))

        # Magnitude of median residual
        rms = float(np.sqrt((median_res**2).mean()))

        # Fraction of pixels with |residual| > 3 (likely watermark signal, not noise)
        strong_pct = float((np.abs(median_res) > 3).mean())

        print(f"  {wm} ({den_name}): NCC={ncc:.3f}  sign_consist={sign_consist:.3f}  "
              f"rms={rms:.3f}  strong_pct={strong_pct:.3f}")

if __name__ == "__main__":
    run_stable_sig()
    run_rivagan()
    analyze_residual_bits()
    print("\n=== done ===")
