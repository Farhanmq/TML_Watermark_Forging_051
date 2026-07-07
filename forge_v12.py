#!/usr/bin/env python3

import sys, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch

CATS = [("WM_1",1,25),("WM_2",26,50),("WM_3",51,75),("WM_4",76,100),
        ("WM_5",101,125),("WM_6",126,150),("WM_7",151,175),("WM_8",176,200)]

RESIDUAL_CFG = {
    "WM_3": ("nlm5",   True,  2.5),
    "WM_4": ("median3",True,  2.5),
    "WM_5": ("wavelet",True,  2.5),
    "WM_6": ("bilat",  True,  2.5),
}

def load_rgb(p): return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
def save_rgb(a, p): Image.fromarray(np.clip(a,0,255).astype(np.uint8)).save(p)
def load_bgr(p): return cv2.imread(str(p))

def dn(name, im):
    u8 = np.clip(im,0,255).astype(np.uint8)
    if name == "nlm5":    return cv2.fastNlMeansDenoisingColored(u8,None,5,5,7,21).astype(np.float32)
    if name == "median3": return cv2.medianBlur(u8,3).astype(np.float32)
    if name == "bilat":   return cv2.bilateralFilter(u8,5,50,50).astype(np.float32)
    if name == "wavelet":
        from skimage.restoration import denoise_wavelet
        return (denoise_wavelet(im/255.,channel_axis=-1,rescale_sigma=True,
                                method="BayesShrink",mode="soft")*255).astype(np.float32)
    raise ValueError(name)

def payload_weighted(imgs, denoiser, reliability, target_rms=2.5):
    res = np.stack([im - dn(denoiser, im) for im in imgs])
    A = np.median(res[0::2], axis=0)
    B = np.median(res[1::2], axis=0)
    payload = np.median(res, axis=0)
    if reliability:
        payload = payload * (np.sign(A) == np.sign(B))
    rms = float(np.sqrt((payload**2).mean())) + 1e-9
    alpha = float(np.clip(target_rms / rms, 1.0, 8.0))
    return payload, alpha

def extract_imwm_msg(wm_name, nbits, method):
    from imwatermark import WatermarkDecoder
    paths = sorted((Path("watermarked_sources")/wm_name).glob("*.png"))
    dec = WatermarkDecoder("bits", nbits)
    rows = []
    for p in paths:
        try:
            rows.append(np.array(dec.decode(load_bgr(p), method), dtype=int))
        except Exception:
            pass
    if not rows: raise RuntimeError(f"Failed {wm_name}/{method}")
    return (np.stack(rows).mean(0) >= 0.5).astype(int).tolist()

def tm_decode_secret(tm, wm_name):
    paths = sorted((Path("watermarked_sources")/wm_name).glob("*.png"))
    secrets = []
    for p in paths:
        try:
            secret, present, _ = tm.decode(Image.open(p).convert("RGB"), MODE="binary")
            if present and secret:
                secrets.append(secret)
        except Exception:
            pass
    if not secrets:
        raise RuntimeError(f"TrustMark decode failed for {wm_name}")
    arr = np.array([[int(b) for b in s] for s in secrets])
    return "".join(map(str, (arr.mean(0) >= 0.5).astype(int)))

def tm_decode_secret_no_ecc(tm, wm_name):
    paths = sorted((Path("watermarked_sources")/wm_name).glob("*.png"))
    secrets = []
    for p in paths:
        try:
            secret, present, _ = tm.decode(Image.open(p).convert("RGB"), MODE="binary")
            if secret:
                secrets.append(secret)
        except Exception:
            pass
    if not secrets:
        raise RuntimeError(f"TrustMark decode failed for {wm_name}")
    arr = np.array([[int(b) for b in s] for s in secrets])
    maj = (arr.mean(0) >= 0.5).astype(int)
    agree = float((arr == maj).mean())
    print(f"  {wm_name} TrustMark-P decode: n={len(secrets)} agree={agree:.3f}")
    return "".join(map(str, maj))

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")

    import lpips
    lpfn = lpips.LPIPS(net="alex").to(dev)
    def to_t(a):
        arr = np.asarray(a, dtype=np.float32)
        if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
        return (torch.from_numpy(arr.transpose(2,0,1))/127.5-1).unsqueeze(0).to(dev)

    from trustmark import TrustMark
    tm_Q = TrustMark(verbose=False, model_type="Q")
    secret7 = tm_decode_secret(tm_Q, "WM_7")
    print(f"WM_7 TrustMark-Q secret len={len(secret7)}")

    tm_P = TrustMark(use_ECC=False, verbose=False, model_type="P")
    secret8 = tm_decode_secret_no_ecc(tm_P, "WM_8")
    print(f"WM_8 TrustMark-P secret len={len(secret8)}")

    from imwatermark import WatermarkDecoder, WatermarkEncoder
    WatermarkDecoder.loadModel()
    WatermarkEncoder.loadModel()
    msg_wm1 = extract_imwm_msg("WM_1", 16, "dwtDct")
    print(f"WM_1 dwtDct msg={''.join(map(str,msg_wm1))}")
    msg_wm2 = extract_imwm_msg("WM_2", 32, "rivaGan")
    print(f"WM_2 RivaGAN msg={''.join(map(str,msg_wm2))}")

    clean = {i: load_rgb(Path("clean_targets")/f"{i}.png") for i in range(1,201)}
    src = lambda wm: [load_rgb(p) for p in sorted((Path("watermarked_sources")/wm).glob("*.png"))]
    stage = Path("stage_v12"); stage.mkdir(exist_ok=True)
    lp = []

    for wm, start, stop in CATS:
        if wm == "WM_7":
            for idx in range(start, stop+1):
                cover = Image.fromarray(np.clip(clean[idx],0,255).astype(np.uint8))
                stego = tm_Q.encode(cover, secret7, MODE="binary")
                stego.save(stage/f"{idx}.png")
                with torch.no_grad(): lp.append(lpfn(to_t(cover), to_t(stego)).item())
            print(f"WM_7: TrustMark-Q native  done")
            continue

        if wm == "WM_8":
            folder_lp = []
            for idx in range(start, stop+1):
                cover = Image.fromarray(np.clip(clean[idx],0,255).astype(np.uint8))
                stego = tm_P.encode(cover, secret8, MODE="binary")
                stego.save(stage/f"{idx}.png")
                with torch.no_grad():
                    l = lpfn(to_t(cover), to_t(stego)).item()
                    lp.append(l); folder_lp.append(l)
            print(f"WM_8: TrustMark-P native  mean_LPIPS={np.mean(folder_lp):.4f}")
            continue

        if wm == "WM_1":
            enc1 = WatermarkEncoder(); enc1.set_watermark("bits", msg_wm1)
            for idx in range(start, stop+1):
                bgr = cv2.cvtColor(np.clip(clean[idx],0,255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                bgr_s = enc1.encode(bgr, "dwtDct")
                rgb_s = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2RGB)
                save_rgb(rgb_s, stage/f"{idx}.png")
                with torch.no_grad(): lp.append(lpfn(to_t(clean[idx]), to_t(rgb_s)).item())
            print(f"WM_1: dwtDct native  done")
            continue

        if wm == "WM_2":
            enc2 = WatermarkEncoder(); enc2.set_watermark("bits", msg_wm2)
            for idx in range(start, stop+1):
                bgr = cv2.cvtColor(np.clip(clean[idx],0,255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                bgr_s = enc2.encode(bgr, "rivaGan")
                rgb_s = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2RGB)
                save_rgb(rgb_s, stage/f"{idx}.png")
                with torch.no_grad(): lp.append(lpfn(to_t(clean[idx]), to_t(rgb_s)).item())
            print(f"WM_2: RivaGAN native  done")
            continue

        den, use_reliab, target_rms = RESIDUAL_CFG[wm]
        payload, alpha = payload_weighted(src(wm), den, use_reliab, target_rms)
        rms = float(np.sqrt((payload**2).mean())) + 1e-9
        folder_lp = []
        for idx in range(start, stop+1):
            forged = np.clip(clean[idx] + alpha*payload, 0, 255)
            save_rgb(forged, stage/f"{idx}.png")
            with torch.no_grad():
                l = lpfn(to_t(clean[idx]), to_t(forged)).item()
                lp.append(l); folder_lp.append(l)
        print(f"{wm}: residual {den} reliab={use_reliab} target_rms={target_rms} "
              f"alpha={alpha:.2f} rms={rms:.3f}  mean_LPIPS={np.mean(folder_lp):.4f}")

    out = "submission_v12.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(1,201): zf.write(stage/f"{i}.png", arcname=f"{i}.png")
    print(f"mean LPIPS={np.mean(lp):.4f}  S_qlt={np.exp(-8*np.mean(lp)):.3f}  wrote {out}")

if __name__ == "__main__":
    main()
