# Assignment 4 — Watermark Forgery

## Task

Black-box watermark forgery: given 25 watermarked source images per folder, inject the same watermark into 200 clean target images without access to the detector or codec.

**Folders:** WM_1 through WM_8, mapping to target images 1–200 (25 each).

## Approach

| Folder | Codec | Method |
|--------|-------|--------|
| WM_1 | dwtDct (16-bit) | Native re-embed via imwatermark |
| WM_2 | RivaGAN (32-bit) | Native re-embed via imwatermark |
| WM_3 | Unknown | Residual averaging (nlm5, rms=2.5) |
| WM_4 | Unknown | Residual averaging (median3, rms=2.5) |
| WM_5 | Unknown | Residual averaging (wavelet, rms=2.5) |
| WM_6 | Unknown | Residual averaging (bilat, rms=2.5) |
| WM_7 | TrustMark-Q | Native re-embed |
| WM_8 | TrustMark-P (no ECC) | Native re-embed |

For unknown codecs, the watermark signal is recovered by computing per-pixel residuals (`image − denoise(image)`) across all 25 sources, taking the median, masking out sign-inconsistent pixels, and injecting the scaled result into each clean target.

## Directory Structure

```
Assignment4/
├── clean_targets/        # 200 clean PNG images (1.png – 200.png)
├── watermarked_sources/
│   ├── WM_1/             # 25 watermarked source PNGs per folder
│   ├── WM_2/
│   └── ...
├── forge_v12.py
├── forge_v7.py
├── sweep_v2.py – sweep_v5.py
└── run_v12.sh
```

## Setup

```bash
pip install "numpy<2.0.0" "opencv-python-headless==4.8.1.78" \
    "huggingface_hub==0.24.6" trustmark scikit-image PyWavelets \
    lpips imwatermark omegaconf einops timm
```

Run on CPU (avoids CUDA version mismatches):

```bash
CUDA_VISIBLE_DEVICES="" python forge_v12.py
```

Output: `submission_v12.zip` containing 200 forged PNGs.

## HTCondor

```bash
condor_submit forge_v12.sub
```

Key submit options:
```
docker_image = pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
request_CPUs = 4
request_memory = 16G
request_GPUs = 1
requirements = UidDomain == "cs.uni-saarland.de"
+WantGPUHomeMounted = true
+WantScratchMounted = true
```

Set `CUDA_VISIBLE_DEVICES=""` in the run script to force CPU (CUDA 13/12 mismatch on cluster nodes).

## Score

| Version | Change | Server Score |
|---------|--------|--------------|
| v7 | Baseline: WM_1/2/7 native + residual fallback | 0.690 |
| v12 | + WM_8 identified as TrustMark-P (no ECC) | **0.779** |
