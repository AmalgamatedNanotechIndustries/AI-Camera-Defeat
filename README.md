# AI-Camera-Defeat

## Installation
pip install numpy pillow qrcode python-barcode matplotlib torch torchvision

## Example Usage

python3 ai_camera_defeat_patch_generator.py --mode full --output patch.png

python3 ai_camera_defeat_patch_generator.py --mode 4c --output patch_with_qr.png --qr "https://example.com"

python3 ai_camera_defeat_patch_generator.py --mode bw --output patch_with_code.png --barcode "123456789012"

python3 ai_camera_defeat_patch_generator.py --mode full --output adv_patch.png --adv --iterations 100

## New Features

--adv: Enables adversarial optimization.

--iterations: Number of optimization steps (default: 50)

## Research Sources

https://arxiv.org/pdf/2101.06896 (new)

https://pmc.ncbi.nlm.nih.gov/articles/PMC9147429/

https://github.com/inspire-group/adv-patch-paper-list/blob/main/README.md
