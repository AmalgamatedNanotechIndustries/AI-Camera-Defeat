# AI-Camera-Defeat
## Installation
pip install numpy pillow qrcode python-barcode matplotlib
## Example Usage
python ai_camera_defeat_patch_generator.py --mode full --output patch.png
python ai_camera_defeat_patch_generator.py --mode 4c --output patch_with_qr.png --qr "https://example.com"
python ai_camera_defeat_patch_generator.py --mode bw --output patch_with_code.png --barcode "123456789012"
python ai_camera_defeat_patch_generator.py --mode full --output adv_patch.png --adv --iterations 100
## New Features
--adv: Enables adversarial optimization.

--iterations: Number of optimization steps (default: 50).
