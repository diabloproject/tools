import argparse
import os
import time

import bitsandbytes as bnb
import cv2
import numpy as np
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Resize, ConvertImageDtype
from tqdm import tqdm


# ------------------------------
# Dataset: video frames (BGR->RGB) normalized to [0,1]
# ------------------------------


class TorchImageDataset(Dataset):
    def __init__(self, folder, target_size=(1080, 1920)):
        self.paths = sorted(
            [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ],
        )
        self.resize = Resize(target_size)
        self.convert = ConvertImageDtype(torch.float32)  # Converts to [0,1]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = read_image(self.paths[idx])  # shape: [C, H, W], dtype: uint8
        img = self.convert(img)  # [0,1] float32
        img = self.resize(img)  # resized to target
        return img


# ------------------------------
# Model: simple 2D conv autoencoder
# Downsample: stride-2 convolutions
# Upsample: transposed convolutions
# ------------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(
                in_ch, base_ch, kernel_size=3, stride=2,
                padding=1,
            ),  # 1080x1920 -> 540x960
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(
                base_ch, base_ch * 2, kernel_size=3,
                stride=2, padding=1,
            ),  # 540x960 -> 270x480
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(
                base_ch * 2, base_ch * 4, kernel_size=3,
                stride=2, padding=1,
            ),  # 270x480 -> 135x240
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(
                base_ch * 4, base_ch * 8, kernel_size=3, stride=2,
                padding=1,
            ),  # 135x240 -> 68x120 (rounding)
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(inplace=True),
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                base_ch * 8, base_ch * 8,
                kernel_size=3, stride=1, padding=1,
            ),
            nn.ReLU(inplace=True),
        )
        # Decoder
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(
                base_ch * 8, base_ch * 4, kernel_size=4,
                stride=2, padding=1,
            ),  # 68x120 -> 136x240
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                base_ch * 4, base_ch * 2, kernel_size=4,
                stride=2, padding=1,
            ),  # 136x240 -> 272x480
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                base_ch * 2, base_ch, kernel_size=4,
                stride=2, padding=1,
            ),  # 272x480 -> 544x960
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                base_ch, in_ch, kernel_size=4,
                stride=2, padding=1,
            ),  # 544x960 -> 1088x1920
            nn.Sigmoid(),  # output in [0,1]
        )

    def forward(self, x):
        # If input isn’t divisible by 16 in each dimension, we’ll crop after decoding to match
        h, w = x.shape[-2], x.shape[-1]
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        z = self.bottleneck(x4)
        y4 = self.dec4(z)
        y3 = self.dec3(y4)
        y2 = self.dec2(y3)
        y1 = self.dec1(y2)
        # y1 may be slightly larger due to rounding; crop to original size
        y1 = y1[..., :h, :w]
        return y1


import torch
import torch.nn as nn


class Autoencoder1024(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        # Encoder
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 1080x1920 -> 540x960
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 540x960 -> 270x480
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 270x480 -> 135x240
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 135x240 -> 68x120
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 68x120 -> 34x60
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),  # 34x60 -> 17x30
            nn.ReLU(True),
            nn.Conv2d(1024, 2048, 4, stride=2, padding=1),  # 17x30 -> 8x15
            nn.ReLU(True),
        )
        # Flatten + linear to latent
        self.enc_fc = nn.Sequential(
            nn.Linear(2048 * 8 * 15, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            # nn.Linear(1024, 1024),
            # nn.GELU(),
            # nn.Linear(1024, 1024),
            # nn.GELU(),
        )

        # Decoder
        self.dec_fc = nn.Sequential(
            # nn.Linear(1024, 1024),
            # nn.GELU(),
            # nn.Linear(1024, 1024),
            # nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048 * 9 * 16),
            nn.GELU(),
        )
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1),  # 34x60 -> 68x120
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),  # 34x60 -> 68x120
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 34x60 -> 68x120
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 68x120 -> 136x240
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 136x240 -> 272x480
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 272x480 -> 544x960
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # 544x960 -> 1088x1920
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        z = self.enc_fc(h)  # latent vector (batch, 1024)
        h_dec = self.dec_fc(z)
        h_dec = h_dec.view(x.size(0), 2048, 9, 16)
        out = self.dec_conv(h_dec)
        out = out[..., :1080, :1920]  # crop to exact size
        return out, z


# ------------------------------
# Utilities: visualize input/output in realtime
# ------------------------------
def tensor_to_bgr_image(t):
    """
    t: torch tensor [C,H,W] in [0,1], RGB
    returns: np.uint8 BGR image
    """
    t = t.detach().cpu().clamp(0, 1)
    img = (t.cpu().to(torch.float).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


def show_preview(input_tensor, output_tensor, window_name="Preview"):
    inp = tensor_to_bgr_image(input_tensor)
    out = tensor_to_bgr_image(output_tensor)
    # Stack side-by-side
    combo = np.hstack([inp, out])
    cv2.imshow(window_name, combo)
    # Non-blocking small wait; exit preview on 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        return False
    return True


# ------------------------------
# Training
# ------------------------------
def train(
    video_path,
    epochs=5,
    batch_size=2,
    lr=1e-3,
    device="cuda",
    preview=True,
    num_workers=0,
):
    # Dataset and loader
    dataset = TorchImageDataset(video_path)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )

    # model = ConvAutoencoder(in_ch=3, base_ch=32).to(device)
    model = Autoencoder1024().to(torch.bfloat16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    print(
        f"Frames: {len(dataset)} | Epochs: {epochs} | Batch size: {batch_size}",
    )
    print(f"Device: {device} | LR: {lr}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        start_t = time.time()

        for i, batch in tqdm(enumerate(loader, start=1), total=len(loader)):
            # print("Batch")
            batch = batch.to(torch.bfloat16).to(device, non_blocking=True)
            # Forward
            recon, _ = model(batch)
            loss = mse_loss(recon, batch)

            # Backward
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += loss.item()

            # Realtime preview: pick one item from this batch
            if preview:
                idx = 0
                ok = show_preview(
                    batch[idx], recon[idx], window_name="Input | Output (Press 'q' to stop preview)",
                )
                if not ok:
                    preview = False  # stop showing further previews

            if i % 10 == 0:
                avg = running_loss / i
                fps = (i * batch.size(0)) / (time.time() - start_t + 1e-9)
                print(
                    f"Epoch {epoch} | Step {i}/{len(loader)} | Loss {avg:.6f} | Frames/s {fps:.2f}",
                )

        print(
            f"Epoch {epoch} complete | Avg loss: {running_loss / max(1, len(loader)):.6f}",
        )

    # Clean up preview windows
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    return model


# ------------------------------
# Entry
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Conv Autoencoder for 1080x1920 video with realtime preview.",
    )
    p.add_argument(
        "--video", type=str, required=True,
        help="Path to input video file.",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--max-frames", type=int, default=None,
        help="Limit frames loaded for faster experimentation.",
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU training.")
    p.add_argument(
        "--no-preview", action="store_true",
        help="Disable realtime preview.",
    )
    p.add_argument(
        "--workers", type=int, default=0,
        help="DataLoader workers.",
    )
    p.add_argument(
        "--stride", type=int, default=1,
        help="Frame subsampling stride (use >1 for fewer frames).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    _ = train(
        video_path=args.video,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        preview=(not args.no_preview),
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
