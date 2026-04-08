#!/usr/bin/env python3
"""Compress exported BMP tensor images into h265 videos.

Run from the tensor export directory (containing manifest.json and BMP images).
Outputs to .compressed/ subdirectory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress tensor BMP images into h265 videos")
    parser.add_argument("--crf", type=int, default=30, help="CRF quality (0=lossless, 51=worst, default: 30)")
    parser.add_argument("--threads", type=int, default=0, help="Encoding threads (0=auto, default: 0)")
    args = parser.parse_args()

    if not os.path.exists('manifest.json'):
        print("Error: manifest.json not found. Run from the tensor export directory.", file=sys.stderr)
        sys.exit(1)

    with open('manifest.json', encoding='utf-8') as f:
        manifest = json.load(f)

    compressed = '.compressed'
    os.makedirs(compressed, exist_ok=True)

    x265_params = f'crf={args.crf}:keyint=10'
    if args.threads > 0:
        x265_params += f':pools={args.threads}'

    for group in manifest['image_groups']:
        w, h = group['width'], group['height']
        images = group['images']

        video_path = os.path.join(compressed, f'{w}x{h}.mkv')
        list_path = os.path.join(compressed, f'filelist_{w}x{h}.txt')

        # Write concat list with explicit durations
        with open(list_path, 'w', encoding='utf-8') as f:
            for img in images:
                f.write(f"file '../{img}'\n")
                f.write("duration 1\n")

        print(f"Compressing {len(images)} frame(s) ({w}x{h}) -> {video_path}")
        subprocess.run([
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0', '-i', list_path,
            '-r', '1',
            '-c:v', 'libx265', '-preset', 'veryslow',
            '-x265-params', x265_params,
            '-pix_fmt', 'gray', video_path,
        ], check=True)

    # Copy manifest into .compressed so decompress.py can work standalone
    shutil.copy2('manifest.json', os.path.join(compressed, 'manifest.json'))

    print(f"Done. Compressed output in {compressed}/")
    print(f"To reconstruct: cd {compressed} && python -m gguf.scripts.gguf_decompress")


if __name__ == '__main__':
    main()
