#!/usr/bin/env python3
"""Export GGUF tensor layers as BMP images with manifest for video compression."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt

from gguf import GGUFReader, GGUFValueType, GGUFWriter, ReaderTensor, dequantize
from gguf.constants import GGMLQuantizationType

logger = logging.getLogger("gguf-tensor-to-image")

# Float types that get RGB encoding (sign/exp/mantissa)
FLOAT_TYPES = {
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.BF16,
}


# --- BMP writers ---

def write_bmp_gray(pixels: npt.NDArray[np.uint8], filepath: str) -> None:
    height, width = pixels.shape
    row_stride = (width + 3) & ~3
    palette_size = 256 * 4
    header_size = 14 + 40
    file_size = header_size + palette_size + row_stride * height

    with open(filepath, 'wb') as f:
        f.write(struct.pack('<2sIHHI', b'BM', file_size, 0, 0, header_size + palette_size))
        f.write(struct.pack('<IiiHHIIiiII', 40, width, height, 1, 8, 0, 0, 0, 0, 256, 256))
        for i in range(256):
            f.write(struct.pack('BBBB', i, i, i, 0))
        pad = b'\x00' * (row_stride - width)
        for y in range(height - 1, -1, -1):
            f.write(pixels[y].tobytes())
            if pad:
                f.write(pad)


def write_bmp_rgb(pixels: npt.NDArray[np.uint8], filepath: str) -> None:
    """Write (H, W, 3) uint8 array as 24-bit BMP. Channel order: R, G, B."""
    height, width, _ = pixels.shape
    row_stride = (width * 3 + 3) & ~3
    header_size = 14 + 40
    file_size = header_size + row_stride * height

    with open(filepath, 'wb') as f:
        f.write(struct.pack('<2sIHHI', b'BM', file_size, 0, 0, header_size))
        f.write(struct.pack('<IiiHHIIiiII', 40, width, height, 1, 24, 0, 0, 0, 0, 0, 0))
        pad = b'\x00' * (row_stride - width * 3)
        for y in range(height - 1, -1, -1):
            bgr = pixels[y, :, ::-1]  # RGB -> BGR for BMP
            f.write(bgr.tobytes())
            if pad:
                f.write(pad)


# --- Float32 <-> RGB ---

def float32_to_rgb(data: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """Convert 2D float32 to (H, W, 3) uint8: R=mantissa_top8, G=exponent, B=sign."""
    bits = data.view(np.uint32)
    sign = ((bits >> 31) & 1).astype(np.uint8) * 255
    exp = ((bits >> 23) & 0xFF).astype(np.uint8)
    mantissa = ((bits >> 15) & 0xFF).astype(np.uint8)
    return np.stack([mantissa, exp, sign], axis=-1)


# --- Tensor processing ---

def linearize_to_uint8(data: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.uint8], float, float]:
    """pixel = (v - min) / (max - min) * 255"""
    vmin = float(data.min())
    vmax = float(data.max())
    if vmax - vmin < 1e-10:
        return np.zeros(data.shape, dtype=np.uint8), vmin, vmax
    return ((data - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8), vmin, vmax


def wrap_1d(data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Wrap 1D data to 32-wide image, pad to nearest 32-row block by repeating last row/col."""
    n = len(data)
    width = 32
    raw_height = -(-n // width)
    padded_height = -(-raw_height // 32) * 32

    row_data = np.full(raw_height * width, data[-1], dtype=np.float32)
    row_data[:n] = data
    img = row_data.reshape(raw_height, width)

    if raw_height < padded_height:
        img = np.vstack([img, np.tile(img[-1:], (padded_height - raw_height, 1))])

    return img


def orient_landscape_2d(arr: npt.NDArray) -> tuple[npt.NDArray, bool]:
    """Transpose 2D array so width >= height."""
    h, w = arr.shape[:2]
    if h > w:
        if arr.ndim == 3:
            return arr.transpose(1, 0, 2), True
        return arr.T, True
    return arr, False


def rewrap_to_32(arr: npt.NDArray) -> npt.NDArray:
    """If either dim < 32, flatten and wrap to 32-wide. Otherwise pad to 32-aligned.
    Works with 2D (H,W) or 3D (H,W,C) arrays."""
    h, w = arr.shape[:2]
    has_channels = arr.ndim == 3
    channels = arr.shape[2] if has_channels else 1

    if h < 32 or w < 32:
        if has_channels:
            flat = arr.reshape(-1, channels)
        else:
            flat = arr.flatten()
        n_pixels = h * w
        width = 32
        raw_h = -(-n_pixels // width)
        pad_h = -(-raw_h // 32) * 32

        if has_channels:
            out = np.tile(flat[-1:], (raw_h * width, 1))
            out[:n_pixels] = flat
            img = out.reshape(raw_h, width, channels)
        else:
            out = np.full(raw_h * width, flat[-1], dtype=arr.dtype)
            out[:n_pixels] = flat
            img = out.reshape(raw_h, width)

        if raw_h < pad_h:
            img = np.concatenate([img, np.tile(img[-1:], (pad_h - raw_h,) + (1,) * (img.ndim - 1))], axis=0)
        return img

    ph = -(-h // 32) * 32
    pw = -(-w // 32) * 32
    if ph == h and pw == w:
        return arr
    if has_channels:
        out = np.empty((ph, pw, channels), dtype=arr.dtype)
    else:
        out = np.empty((ph, pw), dtype=arr.dtype)
    out[:h, :w] = arr
    if pw > w:
        out[:h, w:] = arr[:, -1:]
    if ph > h:
        out[h:] = out[h - 1:h]
    return out


MAX_DIM = 4096


def tile_oversized(arr: npt.NDArray) -> tuple[list[npt.NDArray], int, int]:
    """Split into frames of <= MAX_DIM x MAX_DIM. Works with 2D or 3D arrays."""
    h, w = arr.shape[:2]
    if h <= MAX_DIM and w <= MAX_DIM:
        return [arr], w, h

    tile_w = min(w, MAX_DIM)
    tile_h = min(h, MAX_DIM)
    has_channels = arr.ndim == 3
    channels = arr.shape[2] if has_channels else 1

    if has_channels:
        flat = arr.reshape(-1, channels)
        ppt = tile_w * tile_h
        n_tiles = -(-flat.shape[0] // ppt)
        padded = np.tile(flat[-1:], (n_tiles * ppt, 1))
        padded[:flat.shape[0]] = flat
        tiles = [padded[i * ppt:(i + 1) * ppt].reshape(tile_h, tile_w, channels) for i in range(n_tiles)]
    else:
        flat = arr.flatten()
        ppt = tile_w * tile_h
        n_tiles = -(-len(flat) // ppt)
        padded = np.full(n_tiles * ppt, flat[-1], dtype=arr.dtype)
        padded[:len(flat)] = flat
        tiles = [padded[i * ppt:(i + 1) * ppt].reshape(tile_h, tile_w) for i in range(n_tiles)]

    return tiles, tile_w, tile_h


def get_dequantized(tensor: ReaderTensor) -> npt.NDArray[np.float32]:
    data = dequantize(tensor.data, tensor.tensor_type)
    shape = tuple(int(d) for d in tensor.shape)
    return data.reshape(shape).astype(np.float32)


def _is_float_type(tensor: ReaderTensor) -> bool:
    return tensor.tensor_type in FLOAT_TYPES


# --- Frame writing ---

def _write_frames_gray(frames: list[npt.NDArray[np.uint8]], output_dir: str, layer_id: int) -> None:
    if len(frames) == 1:
        write_bmp_gray(frames[0], os.path.join(output_dir, f"{layer_id:06d}.bmp"))
    else:
        frame_dir = os.path.join(output_dir, f"{layer_id:06d}")
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            write_bmp_gray(frame, os.path.join(frame_dir, f"{i:06d}.bmp"))


def _write_frames_rgb(frames: list[npt.NDArray[np.uint8]], output_dir: str, layer_id: int) -> None:
    if len(frames) == 1:
        write_bmp_rgb(frames[0], os.path.join(output_dir, f"{layer_id:06d}.bmp"))
    else:
        frame_dir = os.path.join(output_dir, f"{layer_id:06d}")
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            write_bmp_rgb(frame, os.path.join(frame_dir, f"{i:06d}.bmp"))


# --- Export ---

def _export_gray(data_2d: npt.NDArray[np.float32], output_dir: str, layer_id: int, entry: dict) -> None:
    """Export a 2D float array as grayscale BMP(s) with min/max linearization."""
    pixels, vmin, vmax = linearize_to_uint8(data_2d)
    pixels, rotated = orient_landscape_2d(pixels)
    pixels = rewrap_to_32(pixels)
    pre_h, pre_w = pixels.shape
    frames, tile_w, tile_h = tile_oversized(pixels)
    _write_frames_gray(frames, output_dir, layer_id)

    entry.update(vmin=vmin, vmax=vmax, rotated=rotated, color="gray",
                 image_width=tile_w, image_height=tile_h,
                 n_frames=len(frames), tiled=len(frames) > 1)
    if len(frames) > 1:
        entry.update(pre_tile_width=pre_w, pre_tile_height=pre_h)


def _export_rgb(data_2d: npt.NDArray[np.float32], output_dir: str, layer_id: int, entry: dict) -> None:
    """Export a 2D float array as RGB BMP(s) encoding sign/exp/mantissa."""
    rgb = float32_to_rgb(data_2d)  # (H, W, 3) uint8
    rgb, rotated = orient_landscape_2d(rgb)
    rgb = rewrap_to_32(rgb)
    pre_h, pre_w = rgb.shape[:2]
    frames, tile_w, tile_h = tile_oversized(rgb)
    _write_frames_rgb(frames, output_dir, layer_id)

    entry.update(rotated=rotated, color="rgb",
                 image_width=tile_w, image_height=tile_h,
                 n_frames=len(frames), tiled=len(frames) > 1)
    if len(frames) > 1:
        entry.update(pre_tile_width=pre_w, pre_tile_height=pre_h)


def export_tensor(tensor: ReaderTensor, output_dir: str, layer_id: int) -> dict:
    """Export a single tensor as BMP image(s). Returns manifest entry."""
    data = get_dequantized(tensor)
    ndim = data.ndim
    use_rgb = _is_float_type(tensor)
    entry = {
        "name": tensor.name,
        "shape": [int(d) for d in tensor.shape],
        "tensor_type": tensor.tensor_type.name,
        "n_elements": int(tensor.n_elements),
        "layer_id": layer_id,
    }

    if ndim <= 2:
        img_data = wrap_1d(data) if ndim == 1 else data
        if use_rgb:
            _export_rgb(img_data, output_dir, layer_id, entry)
        else:
            _export_gray(img_data, output_dir, layer_id, entry)
        tag = entry["color"]
        nf = entry["n_frames"]
        tw, th = entry["image_width"], entry["image_height"]
        logger.info(f"  -> {layer_id:06d}{'/' if nf > 1 else '.bmp'}"
                     f"  ({tw}x{th}, {nf} frame(s), {tag})")

    else:
        rows, cols = data.shape[-2], data.shape[-1]
        slices = data.reshape(-1, rows, cols)
        n_slices = slices.shape[0]

        if use_rgb:
            # RGB: encode each slice's float bits directly
            all_frames = []
            rotated = rows > cols
            for i in range(n_slices):
                rgb = float32_to_rgb(slices[i])
                if rotated:
                    rgb = rgb.transpose(1, 0, 2)
                rgb = rewrap_to_32(rgb)
                all_frames.append(rgb)
            _write_frames_rgb(all_frames, output_dir, layer_id)
            h, w = all_frames[0].shape[:2]
            entry.update(rotated=rotated, color="rgb",
                         image_width=w, image_height=h,
                         n_frames=n_slices, tiled=False)
        else:
            # Gray: per-frame min/max linearization
            rotated = rows > cols
            vmin_list = []
            vmax_list = []
            all_frames = []
            for i in range(n_slices):
                pixels, sv_min, sv_max = linearize_to_uint8(slices[i])
                vmin_list.append(sv_min)
                vmax_list.append(sv_max)
                if rotated:
                    pixels = pixels.T
                pixels = rewrap_to_32(pixels)
                all_frames.append(pixels)
            _write_frames_gray(all_frames, output_dir, layer_id)
            h, w = all_frames[0].shape
            entry.update(vmin=vmin_list, vmax=vmax_list, rotated=rotated, color="gray",
                         image_width=w, image_height=h,
                         n_frames=n_slices, tiled=False)

        logger.info(f"  -> {layer_id:06d}/  ({n_slices} slices, "
                     f"{entry['image_width']}x{entry['image_height']}, {entry['color']})")

    return entry


# --- Metadata serialization ---

def get_architecture(reader: GGUFReader) -> str:
    for name, field in reader.fields.items():
        if name == "general.architecture":
            return str(field.contents())
    return ""


def save_metadata_gguf(reader: GGUFReader, arch: str, output_path: str) -> None:
    """Save all metadata as a tensor-less GGUF file (lossless binary round-trip)."""
    writer = GGUFWriter(output_path, arch)
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
            logger.warning(f"Skipping metadata '{field.name}': {e}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    logger.info(f'* Written {output_path}')


# --- Image grouping ---

def build_image_groups(tensor_manifests: list[dict]) -> list[dict]:
    """Group all images by (width, height, color) for video encoding."""
    groups: dict[tuple[int, int, str], list[str]] = defaultdict(list)
    for t in tensor_manifests:
        lid = t["layer_id"]
        w, h = t["image_width"], t["image_height"]
        color = t.get("color", "gray")
        if t["n_frames"] == 1:
            groups[(w, h, color)].append(f"{lid:06d}.bmp")
        else:
            for s in range(t["n_frames"]):
                groups[(w, h, color)].append(f"{lid:06d}/{s:06d}.bmp")

    return [{"width": w, "height": h, "color": c, "images": imgs}
            for (w, h, c), imgs in sorted(groups.items())]


# --- Main ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Export GGUF tensor layers as BMP images")
    parser.add_argument("model",           type=str,            help="GGUF format model filename")
    parser.add_argument("--output",        type=str,            help="Output directory (default: <model>.tensors/)", default=None)
    parser.add_argument("--filter",        type=str,            help="Regex filter for tensor names", default=None)
    parser.add_argument("--tensor-index",  type=int, nargs='+', help="Export only these tensor indices (0-based)", default=None)
    parser.add_argument("--list",          action="store_true", help="List tensors and exit without exporting")
    parser.add_argument("--verbose",       action="store_true", help="Increase output verbosity")

    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info(f'* Loading: {args.model}')
    reader = GGUFReader(args.model, 'r')
    logger.info(f'* Found {len(reader.tensors)} tensor(s)')

    name_filter = re.compile(args.filter) if args.filter else None

    if args.list:
        for i, tensor in enumerate(reader.tensors):
            shape_str = 'x'.join(str(d) for d in tensor.shape)
            print(f"  {i:4d}: {tensor.tensor_type.name:8s} {shape_str:>20s}  {tensor.name}")  # noqa: NP100
        return

    output_dir = args.output or (args.model + '.tensors')
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f'* Output directory: {output_dir}')

    arch = get_architecture(reader)
    logger.info(f'* Architecture: {arch}')
    save_metadata_gguf(reader, arch, os.path.join(output_dir, 'metadata.gguf'))

    tensor_manifests = []
    for i, tensor in enumerate(reader.tensors):
        if args.tensor_index is not None and i not in args.tensor_index:
            continue
        if name_filter and not name_filter.search(tensor.name):
            continue

        shape_str = 'x'.join(str(d) for d in tensor.shape)
        logger.info(f'[{i:4d}] {tensor.name}  ({tensor.tensor_type.name}, {shape_str})')

        try:
            entry = export_tensor(tensor, output_dir, i)
            tensor_manifests.append(entry)
        except NotImplementedError as e:
            logger.warning(f'  Skipped: {e}')
        except Exception as e:
            logger.error(f'  Error: {e}')
            if args.verbose:
                import traceback
                traceback.print_exc()

    image_groups = build_image_groups(tensor_manifests)

    manifest = {
        "architecture": arch,
        "tensors": tensor_manifests,
        "image_groups": image_groups,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False)
    logger.info(f'* Written {manifest_path}')

    compressed_dir = os.path.join(output_dir, '.compressed')
    os.makedirs(compressed_dir, exist_ok=True)

    logger.info('* Done. Next steps:')
    logger.info(f'  1. cd "{output_dir}" && python -m gguf.scripts.gguf_compress')
    logger.info(f'  2. cd ".compressed" && python -m gguf.scripts.gguf_decompress')


if __name__ == '__main__':
    main()
