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

from gguf import GGUFReader, GGUFWriter, GGUFValueType
from gguf.constants import GGMLQuantizationType


# --- BMP readers ---

def read_bmp_gray(path: str) -> np.ndarray:
    """Read 8-bit grayscale BMP, return (H, W) uint8 array."""
    with open(path, 'rb') as f:
        sig = f.read(2)
        assert sig == b'BM', f"Not a BMP: {path}"
        f.read(8)
        data_offset = struct.unpack('<I', f.read(4))[0]

        header_size = struct.unpack('<I', f.read(4))[0]
        width = struct.unpack('<i', f.read(4))[0]
        height = struct.unpack('<i', f.read(4))[0]
        f.read(2)  # planes
        bpp = struct.unpack('<H', f.read(2))[0]

        f.seek(data_offset)
        abs_h = abs(height)

        if bpp == 24:
            # RGB BMP - read as RGB, return as gray (shouldn't happen but handle gracefully)
            row_stride = (width * 3 + 3) & ~3
            pixels = np.zeros((abs_h, width), dtype=np.uint8)
            for y in range(abs_h - 1, -1, -1) if height > 0 else range(abs_h):
                row_data = np.frombuffer(f.read(row_stride), dtype=np.uint8)[:width * 3]
                pixels[y] = row_data[::3]  # take just one channel
        else:
            row_stride = (width + 3) & ~3
            pixels = np.zeros((abs_h, width), dtype=np.uint8)
            if height > 0:
                for y in range(abs_h - 1, -1, -1):
                    pixels[y] = np.frombuffer(f.read(row_stride), dtype=np.uint8)[:width]
            else:
                for y in range(abs_h):
                    pixels[y] = np.frombuffer(f.read(row_stride), dtype=np.uint8)[:width]
    return pixels


def read_bmp_rgb(path: str) -> np.ndarray:
    """Read 24-bit RGB BMP, return (H, W, 3) uint8 array in RGB order."""
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
        row_stride = (width * 3 + 3) & ~3
        pixels = np.zeros((abs_h, width, 3), dtype=np.uint8)

        if height > 0:
            for y in range(abs_h - 1, -1, -1):
                row_data = np.frombuffer(f.read(row_stride), dtype=np.uint8)[:width * 3]
                bgr = row_data.reshape(width, 3)
                pixels[y] = bgr[:, ::-1]  # BGR -> RGB
        else:
            for y in range(abs_h):
                row_data = np.frombuffer(f.read(row_stride), dtype=np.uint8)[:width * 3]
                bgr = row_data.reshape(width, 3)
                pixels[y] = bgr[:, ::-1]
    return pixels


# --- Float32 reconstruction from RGB ---

def rgb_to_float32(rgb: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) uint8 back to float32. R=mantissa_top8, G=exponent, B=sign."""
    mantissa = rgb[..., 0].astype(np.uint32)
    exp = rgb[..., 1].astype(np.uint32)
    sign = (rgb[..., 2] > 127).astype(np.uint32)
    bits = (sign << 31) | (exp << 23) | (mantissa << 15)
    return bits.view(np.float32)


# --- Tensor reconstruction ---

def delinearize(px: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """v = pixel/255 * (max - min) + min"""
    return px.astype(np.float32) / 255.0 * (vmax - vmin) + vmin


def _read_frames(layer_id: int, n_frames: int, uncompressed: str, color: str) -> list[np.ndarray]:
    """Read BMP frames for a layer."""
    read_fn = read_bmp_rgb if color == 'rgb' else read_bmp_gray
    if n_frames == 1:
        return [read_fn(os.path.join(uncompressed, f"{layer_id:06d}.bmp"))]
    return [read_fn(os.path.join(uncompressed, f"{layer_id:06d}", f"{i:06d}.bmp"))
            for i in range(n_frames)]


def _unrotate(arr: np.ndarray) -> np.ndarray:
    """Un-rotate: transpose back, handling both 2D and 3D (H,W,C) arrays."""
    if arr.ndim == 3:
        return arr.transpose(1, 0, 2)
    return arr.T


def reconstruct_tensor(tensor_info: dict, uncompressed: str) -> np.ndarray:
    """Read images and return float32 tensor in original shape."""
    shape = tensor_info['shape']
    n_elements = tensor_info['n_elements']
    rotated = tensor_info['rotated']
    layer_id = tensor_info['layer_id']
    n_frames = tensor_info['n_frames']
    tiled = tensor_info.get('tiled', False)
    color = tensor_info.get('color', 'gray')
    is_rgb = color == 'rgb'

    frames = _read_frames(layer_id, n_frames, uncompressed, color)

    if tiled:
        pre_w = tensor_info['pre_tile_width']
        pre_h = tensor_info['pre_tile_height']
        if is_rgb:
            flat = np.concatenate([f.reshape(-1, 3) for f in frames], axis=0)[:pre_h * pre_w]
            pixels = flat.reshape(pre_h, pre_w, 3)
        else:
            flat = np.concatenate([f.flatten() for f in frames])[:pre_h * pre_w]
            pixels = flat.reshape(pre_h, pre_w)
        if rotated:
            pixels = _unrotate(pixels)
        if is_rgb:
            data = rgb_to_float32(pixels).flatten()[:n_elements]
        else:
            vmin, vmax = float(tensor_info['vmin']), float(tensor_info['vmax'])
            data = delinearize(pixels, vmin, vmax).flatten()[:n_elements]

    elif n_frames == 1:
        pixels = frames[0]
        if rotated:
            pixels = _unrotate(pixels)
        if is_rgb:
            data = rgb_to_float32(pixels).flatten()[:n_elements]
        else:
            vmin, vmax = float(tensor_info['vmin']), float(tensor_info['vmax'])
            data = delinearize(pixels, vmin, vmax).flatten()[:n_elements]

    else:
        # Multiple frames = 3D slices
        elems_per_slice = n_elements // n_frames
        slices = []
        for s, px in enumerate(frames):
            if rotated:
                px = _unrotate(px)
            if is_rgb:
                sl = rgb_to_float32(px).flatten()[:elems_per_slice]
            else:
                vmin = tensor_info['vmin']
                vmax = tensor_info['vmax']
                sv_min = float(vmin[s]) if isinstance(vmin, list) else float(vmin)
                sv_max = float(vmax[s]) if isinstance(vmax, list) else float(vmax)
                sl = delinearize(px, sv_min, sv_max).flatten()[:elems_per_slice]
            slices.append(sl)
        data = np.concatenate(slices)

    return data.reshape(shape)


# --- Metadata copying ---

def copy_metadata(meta_path: str, writer: GGUFWriter) -> None:
    """Copy all metadata from a metadata-only GGUF into the writer (lossless)."""
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

    # Step 1: Extract frames from videos
    for group in manifest['image_groups']:
        w, h = group['width'], group['height']
        color = group.get('color', 'gray')
        images = group['images']

        video_path = f'{w}x{h}_{color}.mkv'
        pix_fmt = 'bgr24' if color == 'rgb' else 'gray'
        tmp_dir = os.path.join(uncompressed, f'_tmp_{w}x{h}_{color}')
        os.makedirs(tmp_dir, exist_ok=True)

        print(f"Extracting {len(images)} frame(s) from {video_path}...")
        subprocess.run([
            'ffmpeg', '-y', '-i', video_path,
            '-fps_mode', 'passthrough',
            '-pix_fmt', pix_fmt,
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

    # Copy metadata from binary GGUF (lossless)
    if os.path.exists('metadata.gguf'):
        copy_metadata('metadata.gguf', writer)
    else:
        print("Warning: metadata.gguf not found, output will have no metadata", file=sys.stderr)

    # Reconstruct and write tensors
    # Manifest stores shapes in GGML order (innermost dim first).
    # GGUFWriter expects numpy order (outermost dim first) and reverses internally.
    for tensor_info in manifest['tensors']:
        name = tensor_info['name']
        ggml_shape = tensor_info['shape']
        np_shape = list(reversed(ggml_shape))
        tensor_type_name = tensor_info['tensor_type']

        print(f"Reconstructing {name} ({tensor_type_name}, {ggml_shape})...")
        data = reconstruct_tensor(tensor_info, uncompressed)

        qtype = GGMLQuantizationType[tensor_type_name]

        if qtype == GGMLQuantizationType.F32:
            writer.add_tensor(name, data.astype(np.float32), raw_shape=np_shape, raw_dtype=qtype)
        elif qtype == GGMLQuantizationType.F16:
            writer.add_tensor(name, data.astype(np.float16), raw_shape=np_shape, raw_dtype=qtype)
        else:
            # Quantized types: try to re-quantize, fall back to F16
            try:
                from gguf.quants import quantize as gguf_quantize
                quantized = gguf_quantize(data.astype(np.float32).flatten(), qtype)
                writer.add_tensor(name, quantized, raw_shape=np_shape, raw_dtype=qtype)
            except (NotImplementedError, Exception) as e:
                print(f"  Can't re-quantize to {tensor_type_name}, using F16: {e}", file=sys.stderr)
                writer.add_tensor(name, data.astype(np.float16), raw_shape=np_shape,
                                  raw_dtype=GGMLQuantizationType.F16)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    print(f"Done. Reconstructed GGUF: {args.output}")


if __name__ == '__main__':
    main()
