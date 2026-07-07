#!/usr/bin/env python3

import zipfile, argparse, sys
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import cv2

CATS = [("WM_1",1,25),("WM_2",26,50),("WM_3",51,75),("WM_4",76,100),
        ("WM_5",101,125),("WM_6",126,150),("WM_7",151,175),("WM_8",176,200)]
GOOD = {"WM_3":"nlm5","WM_4":"median3","WM_5":"wavelet","WM_6":"bilat"}
HARD = {"WM_8":"nlm5"}

def load_rgb(p): return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
def save_rgb(a,p): Image.fromarray(np.clip(a,0,255).astype(np.uint8)).save(p)
def load_bgr(p): return cv2.imread(str(p))

def dn(name, im):
    u8 = np.clip(im,0,255).astype(np.uint8)
    if name == "nlm5":    return cv2.fastNlMeansDenoisingColored(u8,None,5,5,7,21).astype(np.float32)
    if name == "nlm10":   return cv2.fastNlMeansDenoisingColored(u8,None,10,10,7,21).astype(np.float32)
    if name == "median3": return cv2.medianBlur(u8,3).astype(np.float32)
    if name == "bilat":   return cv2.bilateralFilter(u8,5,50,50).astype(np.float32)
    if name == "wavelet":
        from skimage.restoration import denoise_wavelet
        return (denoise_wavelet(im/255.,channel_axis=-1,rescale_sigma=True,
                                method="BayesShrink",mode="soft")*255).astype(np.float32)
    raise ValueError(name)

def payload_weighted(imgs, denoiser, reliability):
    res = np.stack([im - dn(denoiser, im) for im in imgs])
    A = np.median(res[0::2], axis=0)
    B = np.median(res[1::2], axis=0)
    payload = np.median(res, axis=0)
    if reliability:
        payload = payload * (np.sign(A) == np.sign(B))
    return payload

def extract_imwatermark_msg(wm_name, wm_len, method):
    from imwatermark import WatermarkDecoder
    paths = sorted((Path("watermarked_sources")/wm_name).glob("*.png"))
    dec = WatermarkDecoder("bits", wm_len)
    rows = []
    for p in paths:
        try:
            rows.append(np.array(dec.decode(load_bgr(p), method), dtype=int))
        except Exception:
            pass
    if not rows: raise RuntimeError(f"Failed to extract {wm_name} with {method}")
    return (np.stack(rows).mean(0) >= 0.5).astype(int).tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reliability", action="store_true", default=True)
    ap.add_argument("--target-rms", type=float, default=2.5)
    ap.add_argument("--out", default="submission_v7.zip")
    args = ap.parse_args()

    clean = {i: load_rgb(Path("clean_targets")/f"{i}.png") for i in range(1,201)}
    src = lambda wm: [load_rgb(p) for p in sorted((Path("watermarked_sources")/wm).glob("*.png"))]
    stage = Path("stage_v7"); stage.mkdir(exist_ok=True)

    from trustmark import TrustMark
    tm = TrustMark(verbose=False, model_type="Q")
    wm7 = [Image.open(p).convert("RGB") for p in sorted((Path("watermarked_sources")/"WM_7").glob("*.png"))]
    secret7 = Counter([s for s,pr,_ in (tm.decode(im, MODE="binary") for im in wm7) if pr]).most_common(1)[0][0]
    print(f"WM_7 secret len={len(secret7)}")

    from imwatermark import WatermarkDecoder, WatermarkEncoder
    WatermarkDecoder.loadModel()
    WatermarkEncoder.loadModel()
    msg_wm1 = extract_imwatermark_msg("WM_1", 16, "dwtDct")
    print(f"WM_1 secret: {''.join(map(str, msg_wm1))}")
    msg_wm2 = extract_imwatermark_msg("WM_2", 32, "rivaGan")
    print(f"WM_2 secret: {''.join(map(str, msg_wm2))}")

    import lpips, torch
    dev = "cpu"
    lpfn = lpips.LPIPS(net="alex").to(dev)
    def to_t(a): return (torch.from_numpy(np.asarray(a,dtype=np.float32).transpose(2,0,1))/127.5-1).unsqueeze(0)
    lp = []

    for wm, start, stop in CATS:
        if wm == "WM_7":
            for idx in range(start, stop+1):
                cover = Image.fromarray(np.clip(clean[idx],0,255).astype(np.uint8))
                stego = tm.encode(cover, secret7, MODE="binary")
                stego.save(stage/f"{idx}.png")
                with torch.no_grad(): lp.append(lpfn(to_t(cover), to_t(stego)).item())
            continue

        elif wm == "WM_1":
            enc = WatermarkEncoder()
            enc.set_watermark("bits", msg_wm1)
            for idx in range(start, stop+1):
                bgr_cover = cv2.cvtColor(np.clip(clean[idx],0,255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                bgr_stego = enc.encode(bgr_cover, "dwtDct")
                rgb_stego = cv2.cvtColor(bgr_stego, cv2.COLOR_BGR2RGB)
                save_rgb(rgb_stego, stage/f"{idx}.png")
                with torch.no_grad(): lp.append(lpfn(to_t(clean[idx]), to_t(rgb_stego)).item())
            continue

        elif wm == "WM_2":
            enc = WatermarkEncoder()
            enc.set_watermark("bits", msg_wm2)
            for idx in range(start, stop+1):
                bgr_cover = cv2.cvtColor(np.clip(clean[idx],0,255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                bgr_stego = enc.encode(bgr_cover, "rivaGan")
                rgb_stego = cv2.cvtColor(bgr_stego, cv2.COLOR_BGR2RGB)
                save_rgb(rgb_stego, stage/f"{idx}.png")
                with torch.no_grad(): lp.append(lpfn(to_t(clean[idx]), to_t(rgb_stego)).item())
            continue

        den = GOOD.get(wm) or HARD.get(wm)
        payload = payload_weighted(src(wm), den, args.reliability and wm in GOOD)
        rms = float(np.sqrt((payload**2).mean())) + 1e-9
        alpha = float(np.clip(args.target_rms / rms, 1.0, 8.0))
        print(f"{wm}: {den} reliab={args.reliability and wm in GOOD} alpha={alpha:.2f} rms={rms:.3f}")
        for idx in range(start, stop+1):
            forged = np.clip(clean[idx] + alpha*payload, 0, 255)
            save_rgb(forged, stage/f"{idx}.png")
            with torch.no_grad(): lp.append(lpfn(to_t(clean[idx]), to_t(forged.astype(np.float32))).item())

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(1,201): zf.write(stage/f"{i}.png", arcname=f"{i}.png")
    print(f"mean LPIPS={np.mean(lp):.4f}  S_qlt={np.exp(-8*np.mean(lp)):.3f}  wrote {args.out}")

if __name__ == "__main__":
    main()
