import argparse
import os

import cv2
from tqdm import tqdm

import threading
from queue import Queue


def extract_frames(video_path, output_dir, stride=1, resize=(1920, 1080), ext="png"):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Queue to hold frames to be written
    write_queue = Queue(maxsize=100)

    # Writer thread function
    def writer_thread():
        while True:
            item = write_queue.get()
            if item is None:  # Sentinel value to stop thread
                write_queue.task_done()
                break
            out_path, frame = item
            cv2.imwrite(out_path, frame)
            write_queue.task_done()

    # Start writer threads
    num_threads = 64
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=writer_thread)
        t.start()
        threads.append(t)

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
            write_queue.put((out_path, frame.copy()))
            saved += 1
        count += 1
        pipe.update(1)

    cap.release()

    # Signal threads to stop
    for _ in range(num_threads):
        write_queue.put(None)

    # Wait for all writes to complete
    write_queue.join()

    # Wait for threads to finish
    for t in threads:
        t.join()

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
