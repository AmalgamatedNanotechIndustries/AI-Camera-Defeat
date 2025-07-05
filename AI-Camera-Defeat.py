# ai_camera_defeat_patch_generator.py

"""
This script generates adversarial image patches designed to fool AI-enabled cameras,
based on state-of-the-art research. The output image is user-selectable in color mode (full color, 4 color, black & white),
and can optionally include QR code or barcode overlays.

References:
- https://github.com/inspire-group/adv-patch-paper-list/blob/main/README.md
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9147429/

Installation:
    pip install numpy pillow qrcode python-barcode matplotlib torch torchvision

Usage:
    python ai_camera_defeat_patch_generator.py --mode [full|4c|bw] --output patch.png \
        --qr "Optional QR text" --barcode "Optional barcode text" \
        --adv True --iterations 50
"""

import argparse
import numpy as np
from PIL import Image, ImageDraw
import qrcode
import barcode
from barcode.writer import ImageWriter
import io
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import random


def generate_base_patch(mode: str, size: int = 512) -> Image.Image:
    if mode == "full":
        data = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    elif mode == "4c":
        palette = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0]])
        indices = np.random.randint(0, 4, (size, size))
        data = palette[indices]
    elif mode == "bw":
        data = np.random.choice([0, 255], size=(size, size, 1)).repeat(3, axis=2).astype(np.uint8)
    else:
        raise ValueError("Invalid mode: choose 'full', '4c', or 'bw'")
    return Image.fromarray(data)


def overlay_qr(img: Image.Image, text: str) -> Image.Image:
    qr = qrcode.make(text)
    qr = qr.resize((128, 128))
    img.paste(qr, (10, 10))
    return img


def overlay_barcode(img: Image.Image, text: str) -> Image.Image:
    CODE128 = barcode.get_barcode_class('code128')
    bcode = CODE128(text, writer=ImageWriter(), add_checksum=False)
    fp = io.BytesIO()
    bcode.write(fp)
    bcode_img = Image.open(fp).convert("RGB")
    bcode_img = bcode_img.resize((256, 64))
    img.paste(bcode_img, (img.width - 266, img.height - 74))
    return img


def expectation_over_transformation(img: Image.Image) -> Image.Image:
    angle = random.uniform(-30, 30)
    scale = random.uniform(0.8, 1.2)
    brightness = random.uniform(0.7, 1.3)

    transform = T.Compose([
        T.Resize((int(img.size[1] * scale), int(img.size[0] * scale))),
        T.ColorJitter(brightness=(brightness, brightness)),
        T.RandomRotation((angle, angle)),
        T.Resize((224, 224))
    ])
    return transform(img)


def adv_optimize_patch(image: Image.Image, iterations: int = 50) -> Image.Image:
    model = models.resnet18(pretrained=True)
    model.eval()
    base_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    inv_transform = T.Compose([
        T.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        T.ToPILImage()
    ])

    patch = image.copy()
    patch_tensor = base_transform(patch).unsqueeze(0).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([patch_tensor], lr=0.01)
    target_class = torch.tensor([999])  # Toaster or high-index class

    for _ in range(iterations):
        optimizer.zero_grad()

        transformed_img = expectation_over_transformation(inv_transform(patch_tensor.squeeze().detach()))
        transformed_tensor = base_transform(transformed_img).unsqueeze(0)
        transformed_tensor.requires_grad = True

        output = model(transformed_tensor)
        loss = -F.cross_entropy(output, target_class)
        loss.backward()
        optimizer.step()

        patch_tensor.data = torch.clamp(patch_tensor.data, 0, 1)

    optimized_img = inv_transform(patch_tensor.squeeze().detach().cpu())
    return optimized_img


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial-style camera-defeating patches.")
    parser.add_argument('--mode', choices=['full', '4c', 'bw'], required=True, help="Color mode for patch")
    parser.add_argument('--output', required=True, help="Output file name")
    parser.add_argument('--qr', help="Optional QR code text")
    parser.add_argument('--barcode', help="Optional barcode text")
    parser.add_argument('--adv', action='store_true', help="Apply adversarial optimization")
    parser.add_argument('--iterations', type=int, default=50, help="Iterations for adversarial optimization")
    args = parser.parse_args()

    patch = generate_base_patch(args.mode)

    if args.adv:
        print("Applying adversarial optimization with EoT...")
        patch = adv_optimize_patch(patch, iterations=args.iterations)

    if args.qr:
        patch = overlay_qr(patch, args.qr)
    if args.barcode:
        patch = overlay_barcode(patch, args.barcode)

    patch.save(args.output)
    print(f"Patch saved to {args.output}")


if __name__ == '__main__':
    main()
  
