import argparse
import os

import cv2
from tqdm import tqdm


def extract_frames(video_path, output_dir, stride=1, resize=(1920, 1080), ext="png"):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    saved = 0
    pipe = tqdm(total=total)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % stride == 0:
            # if resize:
            #     frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            out_path = os.path.join(output_dir, f"frame_{saved:06d}.{ext}")
            cv2.imwrite(out_path, frame)
            saved += 1
        count += 1
        pipe.update(1)
    cap.release()
    print(f"Extracted {saved} frames to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--ext", type=str, default="png")
    args = parser.parse_args()

    extract_frames(args.video, args.out, stride=args.stride, resize=(args.width, args.height), ext=args.ext)
