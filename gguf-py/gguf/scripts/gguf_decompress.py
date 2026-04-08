#!/usr/bin/env python3
"""Decompress tensor videos/images and reconstruct a GGUF file.

Run from the .compressed/ directory (containing manifest.json and compressed files).
Extracts to .uncompressed/ then writes reconstructed.gguf.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from collections import defaultdict

import numpy as np

from gguf import GGUFWriter, GGUFValueType
from gguf.constants import GGMLQuantizationType


# --- BMP reader ---

def read_bmp(path: str) -> np.ndarray:
    """Read 8-bit grayscale BMP, return (H, W) uint8 array."""
    with open(path, 'rb') as f:
        sig = f.read(2)
        assert sig == b'BM', f"Not a BMP: {path}"
        f.read(8)
        data_offset = struct.unpack('<I', f.read(4))[0]

        header_size = struct.unpack('<I', f.read(4))[0]
        width = struct.unpack('<i', f.read(4))[0]
        height = struct.unpack('<i', f.read(4))[0]

        f.seek(data_offset)
        abs_h = abs(height)
        row_stride = (width + 3) & ~3
        pixels = np.zeros((abs_h, width), dtype=np.uint8)

        if height > 0:  # bottom-up
            for y in range(abs_h - 1, -1, -1):
                pixels[y] = np.frombuffer(f.read(row_stride), dtype=np.uint8)[:width]
        else:  # top-down
            for y in range(abs_h):
                pixels[y] = np.frombuffer(f.read(row_stride), dtype=np.uint8)[:width]
    return pixels


# --- Tensor reconstruction ---

def delinearize(px: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """v = pixel/255 * (max - min) + min"""
    return px.astype(np.float32) / 255.0 * (vmax - vmin) + vmin


def _read_frames(layer_id: int, n_frames: int, uncompressed: str) -> list[np.ndarray]:
    """Read BMP frames for a layer."""
    if n_frames == 1:
        return [read_bmp(os.path.join(uncompressed, f"{layer_id:06d}.bmp"))]
    return [read_bmp(os.path.join(uncompressed, f"{layer_id:06d}", f"{i:06d}.bmp"))
            for i in range(n_frames)]


def reconstruct_tensor(tensor_info: dict, uncompressed: str) -> np.ndarray:
    """Read images and return float32 tensor in original shape."""
    shape = tensor_info['shape']
    n_elements = tensor_info['n_elements']
    vmin = tensor_info['vmin']
    vmax = tensor_info['vmax']
    rotated = tensor_info['rotated']
    layer_id = tensor_info['layer_id']
    n_frames = tensor_info['n_frames']
    tiled = tensor_info.get('tiled', False)
    ndim = len(shape)

    frames = _read_frames(layer_id, n_frames, uncompressed)

    if tiled:
        # Frames are tiles of a single large image -- reassemble
        pre_w = tensor_info['pre_tile_width']
        pre_h = tensor_info['pre_tile_height']
        flat = np.concatenate([f.flatten() for f in frames])[:pre_h * pre_w]
        pixels = flat.reshape(pre_h, pre_w)
        if rotated:
            pixels = pixels.T
        data = delinearize(pixels, float(vmin), float(vmax))
        data = data.flatten()[:n_elements]

    elif n_frames == 1:
        pixels = frames[0]
        if rotated:
            pixels = pixels.T
        data = delinearize(pixels, float(vmin), float(vmax))
        # Truncate padding from rewrap/pad (works for 1D and 2D)
        data = data.flatten()[:n_elements]

    else:
        # Multiple frames are 3D slices, per-frame vmin/vmax
        slices = []
        for s, px in enumerate(frames):
            if rotated:
                px = px.T
            sv_min = float(vmin[s]) if isinstance(vmin, list) else float(vmin)
            sv_max = float(vmax[s]) if isinstance(vmax, list) else float(vmax)
            slices.append(delinearize(px, sv_min, sv_max))
        data = np.stack(slices)

    return data.reshape(shape)


# --- Metadata writing ---

def write_metadata(writer: GGUFWriter, metadata: list[dict]) -> None:
    for entry in metadata:
        key = entry['key']
        type_str = entry['type']
        value = entry['value']

        try:
            if type_str.startswith('ARRAY:'):
                elem_name = type_str[6:]
                vtype = GGUFValueType.ARRAY
                sub_type = GGUFValueType[elem_name]
                if not value:
                    continue
                writer.add_key_value(key, value, vtype, sub_type=sub_type)
            else:
                vtype = GGUFValueType[type_str]
                writer.add_key_value(key, value, vtype)
        except Exception as e:
            print(f"  Warning: skipping metadata '{key}': {e}", file=sys.stderr)


# --- Main ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Decompress tensor images and reconstruct GGUF")
    parser.add_argument("--output", type=str, default="reconstructed.gguf", help="Output GGUF path")
    args = parser.parse_args()

    if not os.path.exists('manifest.json'):
        print("Error: manifest.json not found. Run from the .compressed/ directory.", file=sys.stderr)
        sys.exit(1)

    with open('manifest.json', encoding='utf-8') as f:
        manifest = json.load(f)

    uncompressed = '.uncompressed'
    os.makedirs(uncompressed, exist_ok=True)

    # Step 1: Extract frames from mp4 videos
    for group in manifest['image_groups']:
        w, h = group['width'], group['height']
        images = group['images']

        video_path = f'{w}x{h}.mkv'
        tmp_dir = os.path.join(uncompressed, f'_tmp_{w}x{h}')
        os.makedirs(tmp_dir, exist_ok=True)

        print(f"Extracting {len(images)} frame(s) from {video_path}...")
        subprocess.run([
            'ffmpeg', '-y', '-i', video_path,
            '-fps_mode', 'passthrough',
            '-pix_fmt', 'gray',
            os.path.join(tmp_dir, 'frame_%06d.bmp'),
        ], check=True)

        # Rename frames to original paths
        for idx, img_path in enumerate(images):
            src = os.path.join(tmp_dir, f'frame_{idx + 1:06d}.bmp')
            dst = os.path.join(uncompressed, img_path)
            os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
            shutil.move(src, dst)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Step 2: Reconstruct GGUF
    arch = manifest.get('architecture', '')
    writer = GGUFWriter(args.output, arch)

    # Write metadata
    write_metadata(writer, manifest.get('metadata', []))

    # Reconstruct and write tensors
    for tensor_info in manifest['tensors']:
        name = tensor_info['name']
        shape = tensor_info['shape']
        tensor_type_name = tensor_info['tensor_type']

        print(f"Reconstructing {name} ({tensor_type_name}, {shape})...")
        data = reconstruct_tensor(tensor_info, uncompressed)

        qtype = GGMLQuantizationType[tensor_type_name]

        if qtype == GGMLQuantizationType.F32:
            writer.add_tensor(name, data.astype(np.float32), raw_shape=shape, raw_dtype=qtype)
        elif qtype == GGMLQuantizationType.F16:
            writer.add_tensor(name, data.astype(np.float16), raw_shape=shape, raw_dtype=qtype)
        else:
            # Quantized types: try to re-quantize, fall back to F16
            try:
                from gguf.quants import quantize as gguf_quantize
                quantized = gguf_quantize(data.astype(np.float32).flatten(), qtype)
                writer.add_tensor(name, quantized, raw_shape=shape, raw_dtype=qtype)
            except (NotImplementedError, Exception) as e:
                print(f"  Can't re-quantize to {tensor_type_name}, using F16: {e}", file=sys.stderr)
                writer.add_tensor(name, data.astype(np.float16), raw_shape=shape,
                                  raw_dtype=GGMLQuantizationType.F16)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    print(f"Done. Reconstructed GGUF: {args.output}")


if __name__ == '__main__':
    main()
