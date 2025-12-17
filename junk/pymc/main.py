import argparse
import os
import time

import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
from libraries.python.ml.sequence_datasets.image import ImageDataset


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


class MicroLast(nn.Module):
    def __init__(self, n_overlays: int, latent_size=1024):
        super().__init__()
        self.n_overlays = n_overlays
        self.selectors = nn.ModuleList([nn.Linear(latent_size, 1) for _ in range(n_overlays)])
        self.offsets = nn.Parameter(torch.zeros((n_overlays, 3, 1080, 1920)))
        self.scales = nn.Parameter(torch.ones((n_overlays, 3, 1080, 1920)))

    def forward(self, x, lat):
        for i in range(self.n_overlays):
            selector = F.sigmoid(self.selectors[i](lat))
            selector = selector.view(*selector.shape, 1, 1)
            x = (1 - selector) * x + selector * (self.offsets[i] + self.scales[i] * x)
        return x


class Autoencoder1024(nn.Module):
    def __init__(self, folding_rank=0):
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
            nn.ReLU(True),
        )
        self.last = MicroLast(16)

    def forward(self, x):
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        z = self.enc_fc(h)  # latent vector (batch, 1024)
        h_dec = self.dec_fc(z)
        h_dec = h_dec.view(x.size(0), 2048, 9, 16)
        out = self.dec_conv(h_dec)
        out = out[..., :1080, :1920]  # crop to exact size
        before_microlast = F.sigmoid(out)
        out = self.last(out, z)
        out = F.sigmoid(out)
        return out, before_microlast


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


def show_preview(input_tensor, output_tensor, before_microlast=None, model=None, window_name="Preview"):
    inp = tensor_to_bgr_image(input_tensor)
    out = tensor_to_bgr_image(output_tensor)

    # Stack images: input | before_microlast | output
    if before_microlast is not None:
        before_ml = tensor_to_bgr_image(before_microlast)
        combo = np.hstack([inp, before_ml, out])
    else:
        combo = np.hstack([inp, out])
    cv2.imshow(window_name, combo)

    # Display overlay previews at 360p in a single window
    if model is not None and hasattr(model, 'last'):
        # Create a ones_like template at 360p (640x360)
        preview_h, preview_w = 360, 640
        overlay_images = []

        for i in range(model.last.n_overlays):
            # Get overlay parameters
            offset = model.last.offsets[i].detach().cpu()
            scale = model.last.scales[i].detach().cpu()

            # Create ones_like input
            ones = torch.ones_like(offset)

            # Apply overlay transformation
            overlay_vis = offset + scale * ones

            # Resize to 360p and convert to BGR
            overlay_resized = F.interpolate(
                overlay_vis.unsqueeze(0),
                size=(preview_h, preview_w),
                mode='bilinear',
                align_corners=False,
            )[0]

            # Normalize for visualization
            overlay_img = tensor_to_bgr_image(overlay_resized)
            overlay_images.append(overlay_img)

        # Arrange overlays in a grid (4x4 for 16 overlays)
        rows = []
        for row_idx in range(4):
            row = np.hstack(overlay_images[row_idx * 4:(row_idx + 1) * 4])
            rows.append(row)
        overlays_grid = np.vstack(rows)
        cv2.imshow("Overlays", overlays_grid)

    # Non-blocking small wait; exit preview on 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        return False
    return True


# ------------------------------
# Loss with Sobel for edge preservation
# ------------------------------
def sobel_loss(input_tensor, output_tensor):
    # Convert to grayscale
    input_gray = input_tensor.mean(dim=1, keepdim=True)  # [B,1,H,W]
    output_gray = output_tensor.mean(dim=1, keepdim=True)

    # Sobel kernels
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=input_tensor.dtype, device=input_tensor.device,
    ).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=input_tensor.dtype, device=input_tensor.device,
    ).unsqueeze(0).unsqueeze(0)

    # Compute gradients
    Gx_in = F.conv2d(input_gray, sobel_x, padding=1)
    Gy_in = F.conv2d(input_gray, sobel_y, padding=1)
    mag_in = torch.sqrt(Gx_in ** 2 + Gy_in ** 2 + 1e-6)

    Gx_out = F.conv2d(output_gray, sobel_x, padding=1)
    Gy_out = F.conv2d(output_gray, sobel_y, padding=1)
    mag_out = torch.sqrt(Gx_out ** 2 + Gy_out ** 2 + 1e-6)

    return mse_loss(mag_in, mag_out)



import torch
import torch.nn as nn

class ESPCNx3(nn.Module):
    """
    Efficient Sub-Pixel CNN for 3x upscaling.
    Input:  (B, 3, H, W)  where HxW ~ 360p
    Output: (B, 3, 3H, 3W) where ~1080p
    """
    def __init__(self, channels=3, feat=64, upscale=3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, feat, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat, channels * (upscale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale)  # rearranges C*(r^2) to spatial upscaling by r
        )

    def forward(self, x):
        return self.body(x)



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
    checkpoint_every=None,
):
    # Dataset and loader
    dataset = ImageDataset(video_path)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )

    # model = ConvAutoencoder(in_ch=3, base_ch=32).to(device)
    model = Autoencoder1024().to(torch.bfloat16).to(device)
    opt = torch.optim.AdamW(
        [
            {'params': model.enc_conv.parameters(), 'lr': lr},
            {'params': model.enc_fc.parameters(), 'lr': lr},
            {'params': model.dec_conv.parameters(), 'lr': lr},
            {'params': model.dec_fc.parameters(), 'lr': lr},
            {'params': [model.last.scales], 'lr': lr},
            {'params': [model.last.offsets], 'lr': lr},
            {'params': model.last.selectors.parameters(), 'lr': lr},
        ],
    )
    model = torch.compile(model)
    if checkpoint_every is not None:
        os.makedirs("checkpoints", exist_ok=True)

    print(
        f"Frames: {len(dataset)} | Epochs: {epochs} | Batch size: {batch_size}",
    )
    print(f"Device: {device} | LR: {lr}")
    count = 0
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        start_t = time.time()
        pipe = tqdm(enumerate(loader, start=1), total=len(loader))
        for i, batch in pipe:
            count += 1
            # print("Batch")
            batch = batch.to(torch.bfloat16).to(device, non_blocking=True)
            # Forward
            recon, before_ml = model(batch)
            loss = 0.5 * mse_loss(recon, batch) + sobel_loss(batch, recon)

            # Backward
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_loss += loss.item()

            # Realtime preview: pick one item from this batch
            if preview:
                ok = show_preview(
                    batch[0], recon[0], before_microlast=before_ml[0], model=model,
                    window_name="Input | Before MicroLast | Output (Press 'q' to stop preview)",
                )

            if i % 10 == 0:
                avg = running_loss / i
                fps = (i * batch.size(0)) / (time.time() - start_t + 1e-9)
                pipe.set_description(
                    f"Epoch {epoch} | Step {i}/{len(loader)} | Loss {avg:.6f} | Frames/s {fps:.2f}",
                )

            if checkpoint_every is not None and count % checkpoint_every == 0:
                torch.save(model.state_dict(), f"checkpoints/model_epoch_gstep_{count}.pth")  # type: ignore

        print(f"Epoch {epoch} complete | Avg loss: {running_loss / max(1, len(loader)):.6f}")

    # Clean up preview windows
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    return model



import torch
import torch.nn as nn
import torch.optim as optim
import random
from torchvision.transforms.functional import resize, InterpolationMode

def make_lr_hr_pair(hr, scale=3, patch_hr=192):  # 192x192 HR patch -> 64x64 LR
    _, _, H, W = hr.shape
    y = random.randint(0, H - patch_hr)
    x = random.randint(0, W - patch_hr)
    hr_patch = hr[:, :, y:y+patch_hr, x:x+patch_hr]
    lr_patch = resize(hr_patch, [patch_hr//scale, patch_hr//scale], InterpolationMode.BICUBIC, antialias=True)
    return lr_patch, hr_patch


import cv2
import numpy as np
import torch
import torch.nn.functional as F

def tensor_to_bgr_image(t: torch.Tensor):
    """
    Convert a CHW tensor in [0,1] to BGR uint8 for OpenCV.
    """
    arr = t.detach().to(torch.float32).cpu().clamp(0,1).numpy()
    arr = np.transpose(arr, (1,2,0))  # HWC
    arr = (arr * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def show_preview_upscaler(sr_tensor, hr_tensor=None, window_name="Preview"):
    """
    Show side-by-side preview:
    - LR input (360p)
    - Bicubic baseline (optional, if hr_tensor provided)
    - SR output (1080p)
    """
    # inp = hr_tensor.detach().to(torch.float32).cpu().numpy()
    # inp = np.transpose(inp, (1, 2, 0))
    inp = tensor_to_bgr_image(hr_tensor)
    out = tensor_to_bgr_image(sr_tensor)

    print(inp.shape, out.shape)
    combo = np.hstack([inp, out])

    cv2.imshow(window_name, combo)

    # Non-blocking wait; quit preview on 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        return False
    return True



import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# assume you already defined ESPCNx3 and ImageDataset(video_path)

def train_upscaler(
    video_path,
    epochs=5,
    batch_size=2,
    lr=1e-3,
    device="cuda",
    preview=True,
    num_workers=0,
    checkpoint_every=None,
):
    # Dataset and loader
    dataset = ImageDataset(video_path)  # should yield HR frames
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )

    # Model + optimizer
    model = ESPCNx3().to(torch.bfloat16).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    model = torch.compile(model)
    if checkpoint_every is not None:
        os.makedirs("checkpoints", exist_ok=True)

    print(f"Frames: {len(dataset)} | Epochs: {epochs} | Batch size: {batch_size}")
    print(f"Device: {device} | LR: {lr}")

    count = 0
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        start_t = time.time()
        pipe = tqdm(enumerate(loader, start=1), total=len(loader))
        for i, hr_batch in pipe:
            count += 1
            hr_batch = hr_batch.to(torch.bfloat16).to(device, non_blocking=True)

            # synthesize LR by downscaling HR
            lr_batch = torch.nn.functional.interpolate(
                hr_batch.float(), scale_factor=1/3,
                mode="bicubic", align_corners=False, antialias=True
            ).to(torch.bfloat16)

            # forward
            pred = model(lr_batch)
            loss = loss_fn(pred, hr_batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            pipe.set_description(f"Epoch {epoch} Loss {loss.item():.4f}")

            # optional preview
            if preview:
                # lr_batch: (B,3,360,640), sr_pred: (B,3,1080,1920), hr_batch: (B,3,1080,1920)
                ok = show_preview_upscaler(pred[0], hr_batch[0])
                if not ok:
                    break


            # checkpoint
            if checkpoint_every and count % checkpoint_every == 0:
                ckpt_path = f"checkpoints/upscaler_{count}.pt"
                torch.save(model.state_dict(), ckpt_path)

        dur = time.time() - start_t
        print(f"Epoch {epoch} done in {dur:.1f}s, avg loss={running_loss/len(loader):.4f}")




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
    p.add_argument("--lr", type=float, default=2e-4)
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
    p.add_argument(
        "--checkpoint-every", type=int, default=None,
        help="Save checkpoint every N steps (batches).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    _ = train_upscaler(
        video_path=args.video,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        preview=(not args.no_preview),
        num_workers=args.workers,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
