#!/usr/bin/env python3
"""Decompress tensor videos/bins and reconstruct a GGUF file.

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

import numpy as np

from gguf import GGUFReader, GGUFWriter, GGUFValueType
from gguf.constants import GGMLQuantizationType


# --- BMP reader (8-bit grayscale) ---

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

        if height > 0:
            for y in range(abs_h - 1, -1, -1):
                pixels[y] = np.frombuffer(f.read(row_stride), dtype=np.uint8)[:width]
        else:
            for y in range(abs_h):
                pixels[y] = np.frombuffer(f.read(row_stride), dtype=np.uint8)[:width]
    return pixels


# --- SEM reconstruction ---

def sem_to_float32(sign: np.ndarray, exp: np.ndarray, mantissa: np.ndarray) -> np.ndarray:
    """Reconstruct float32 from sign, exponent, mantissa grayscale images."""
    s = (sign > 127).astype(np.uint32)
    e = exp.astype(np.uint32)
    m = mantissa.astype(np.uint32)
    bits = (s << 31) | (e << 23) | (m << 15)
    return bits.view(np.float32)


# --- Tensor reconstruction ---

def _read_frames(layer_id: int, n_frames: int, uncompressed: str) -> list[np.ndarray]:
    if n_frames == 1:
        return [read_bmp(os.path.join(uncompressed, f"{layer_id:06d}.bmp"))]
    return [read_bmp(os.path.join(uncompressed, f"{layer_id:06d}", f"{i:06d}.bmp"))
            for i in range(n_frames)]


def reconstruct_tensor_sem(tensor_info: dict, uncompressed: str) -> np.ndarray:
    """Reconstruct tensor from SEM (sign/exp/mantissa) grayscale frames."""
    np_shape = list(reversed(tensor_info['shape']))
    n_elements = tensor_info['n_elements']
    rotated = tensor_info['rotated']
    layer_id = tensor_info['layer_id']
    n_frames = tensor_info['n_frames']
    tiled = tensor_info.get('tiled', False)

    frames = _read_frames(layer_id, n_frames, uncompressed)

    # Frames are in groups of 3: [sign, exp, mantissa, sign, exp, mantissa, ...]
    n_logical = n_frames // 3

    if tiled:
        pre_w = tensor_info['pre_tile_width']
        pre_h = tensor_info['pre_tile_height']
        # Reconstruct each tile from its 3 channels, then assemble
        all_data = []
        for t in range(n_logical):
            s, e, m = frames[t * 3], frames[t * 3 + 1], frames[t * 3 + 2]
            all_data.append(sem_to_float32(s, e, m).flatten())
        flat = np.concatenate(all_data)[:pre_h * pre_w]
        data = flat.reshape(pre_h, pre_w)
        if rotated:
            data = data.T
        data = data.flatten()[:n_elements]

    elif n_logical == 1:
        s, e, m = frames[0], frames[1], frames[2]
        data = sem_to_float32(s, e, m)
        if rotated:
            data = data.T
        data = data.flatten()[:n_elements]

    else:
        # Multiple logical frames = 3D slices
        elems_per_slice = n_elements // n_logical
        slices = []
        for i in range(n_logical):
            s, e, m = frames[i * 3], frames[i * 3 + 1], frames[i * 3 + 2]
            sl = sem_to_float32(s, e, m)
            if rotated:
                sl = sl.T
            slices.append(sl.flatten()[:elems_per_slice])
        data = np.concatenate(slices)

    return data.reshape(np_shape)


def reconstruct_tensor_bin(tensor_info: dict) -> np.ndarray:
    layer_id = tensor_info['layer_id']
    np_shape = list(reversed(tensor_info['shape']))
    n_elements = tensor_info['n_elements']

    bin_path = f"{layer_id:06d}.bin"
    data = np.fromfile(bin_path, dtype=np.float32, count=n_elements)
    return data.reshape(np_shape)


# --- Metadata copying ---

def copy_metadata(meta_path: str, writer: GGUFWriter) -> None:
    reader = GGUFReader(meta_path, 'r')
    for field in reader.fields.values():
        if field.name == "general.architecture" or field.name.startswith('GGUF.'):
            continue
        if not field.types:
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        try:
            value = field.contents()
            if value is not None:
                writer.add_key_value(field.name, value, val_type, sub_type=sub_type)
        except Exception as e:
            print(f"  Warning: skipping metadata '{field.name}': {e}", file=sys.stderr)


# --- Main ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Decompress tensor data and reconstruct GGUF")
    parser.add_argument("--output", type=str, default="reconstructed.gguf", help="Output GGUF path")
    args = parser.parse_args()

    if not os.path.exists('manifest.json'):
        print("Error: manifest.json not found. Run from the .compressed/ directory.", file=sys.stderr)
        sys.exit(1)

    with open('manifest.json', encoding='utf-8') as f:
        manifest = json.load(f)

    uncompressed = '.uncompressed'
    os.makedirs(uncompressed, exist_ok=True)

    # Step 1: Extract frames from videos
    for group in manifest['image_groups']:
        w, h = group['width'], group['height']
        ch = group.get('channel', 'gray')
        images = group['images']

        video_path = f'{w}x{h}_{ch}.mkv'
        tmp_dir = os.path.join(uncompressed, f'_tmp_{w}x{h}_{ch}')
        os.makedirs(tmp_dir, exist_ok=True)

        print(f"Extracting {len(images)} frame(s) from {video_path}...")
        subprocess.run([
            'ffmpeg', '-y', '-i', video_path,
            '-fps_mode', 'passthrough',
            '-pix_fmt', 'gray',
            os.path.join(tmp_dir, 'frame_%06d.bmp'),
        ], check=True)

        for idx, img_path in enumerate(images):
            src = os.path.join(tmp_dir, f'frame_{idx + 1:06d}.bmp')
            dst = os.path.join(uncompressed, img_path)
            os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
            shutil.move(src, dst)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Step 2: Reconstruct GGUF
    arch = manifest.get('architecture', '')
    writer = GGUFWriter(args.output, arch)

    if os.path.exists('metadata.gguf'):
        copy_metadata('metadata.gguf', writer)
    else:
        print("Warning: metadata.gguf not found, output will have no metadata", file=sys.stderr)

    for tensor_info in manifest['tensors']:
        name = tensor_info['name']
        ggml_shape = tensor_info['shape']
        np_shape = list(reversed(ggml_shape))
        tensor_type_name = tensor_info['tensor_type']
        encoding = tensor_info.get('encoding', 'sem')

        print(f"Reconstructing {name} ({tensor_type_name}, {ggml_shape}, {encoding})...")

        if encoding == 'bin':
            data = reconstruct_tensor_bin(tensor_info)
        else:
            data = reconstruct_tensor_sem(tensor_info, uncompressed)

        qtype = GGMLQuantizationType[tensor_type_name]

        if encoding == 'bin':
            # Lossless passthrough: write back in original type
            if qtype == GGMLQuantizationType.F32:
                writer.add_tensor(name, data.astype(np.float32), raw_shape=np_shape, raw_dtype=qtype)
            elif qtype == GGMLQuantizationType.F16:
                writer.add_tensor(name, data.astype(np.float16), raw_shape=np_shape, raw_dtype=qtype)
            else:
                writer.add_tensor(name, data.astype(np.float32), raw_shape=np_shape,
                                  raw_dtype=GGMLQuantizationType.F32)
        else:
            # SEM-encoded: write as F16 (SEM gives ~8-bit mantissa, re-quantizing amplifies error)
            writer.add_tensor(name, data.astype(np.float16), raw_shape=np_shape,
                              raw_dtype=GGMLQuantizationType.F16)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    print(f"Done. Reconstructed GGUF: {args.output}")


if __name__ == '__main__':
    main()
