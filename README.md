# Watermark Forgery — Team 051

## How to Recreate the Best Result (score 0.779)

### Prerequisites

Place the data in the same directory as the scripts:
```
clean_targets/        # 200 clean PNG images (1.png – 200.png)
watermarked_sources/
    WM_1/ ... WM_8/   # 25 watermarked source PNGs per folder
```

### Install dependencies

```bash
pip install "numpy<2.0.0" "opencv-python-headless==4.8.1.78" \
    "huggingface_hub==0.24.6" trustmark scikit-image PyWavelets \
    lpips imwatermark omegaconf einops timm
```

### Run

```bash
CUDA_VISIBLE_DEVICES="" python forge_v12.py
```

Output: `submission_v12.zip` containing 200 forged PNGs ready for submission.

### HTCondor (cluster)

```bash
condor_submit forge_v12.sub
```
