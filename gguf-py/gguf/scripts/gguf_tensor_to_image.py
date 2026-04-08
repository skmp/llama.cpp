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

from gguf import GGUFReader, GGUFValueType, ReaderTensor, dequantize

logger = logging.getLogger("gguf-tensor-to-image")


# --- BMP writer ---

def write_bmp(pixels: npt.NDArray[np.uint8], filepath: str) -> None:
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
    raw_height = -(-n // width)  # ceil(n / 32)
    padded_height = -(-raw_height // 32) * 32  # round up to multiple of 32

    # Fill rows, pad partial last row by repeating last value
    row_data = np.full(raw_height * width, data[-1], dtype=np.float32)
    row_data[:n] = data
    img = row_data.reshape(raw_height, width)

    # Pad rows to padded_height by repeating last row
    if raw_height < padded_height:
        img = np.vstack([img, np.tile(img[-1:], (padded_height - raw_height, 1))])

    return img


def orient_landscape(pixels: npt.NDArray[np.uint8]) -> tuple[npt.NDArray[np.uint8], bool]:
    h, w = pixels.shape
    if h > w:
        return pixels.T, True
    return pixels, False


def rewrap_to_32(pixels: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """If either dimension < 32, flatten and wrap to 32-wide, pad to 32-row blocks.
    Otherwise pad both dims to multiples of 32."""
    h, w = pixels.shape

    if h < 32 or w < 32:
        # Flatten and wrap to 32-wide (like 1D treatment)
        flat = pixels.flatten()
        n = len(flat)
        width = 32
        raw_h = -(-n // width)
        pad_h = -(-raw_h // 32) * 32
        out = np.full(raw_h * width, flat[-1], dtype=np.uint8)
        out[:n] = flat
        img = out.reshape(raw_h, width)
        if raw_h < pad_h:
            img = np.vstack([img, np.tile(img[-1:], (pad_h - raw_h, 1))])
        return img

    # Normal padding to 32-aligned
    ph = -(-h // 32) * 32
    pw = -(-w // 32) * 32
    if ph == h and pw == w:
        return pixels
    out = np.empty((ph, pw), dtype=np.uint8)
    out[:h, :w] = pixels
    if pw > w:
        out[:h, w:] = pixels[:, -1:]
    if ph > h:
        out[h:, :] = out[h - 1:h, :]
    return out


MAX_DIM = 4096


def tile_oversized(pixels: npt.NDArray[np.uint8]) -> tuple[list[npt.NDArray[np.uint8]], int, int]:
    """Split image into frames of <= MAX_DIM x MAX_DIM. Returns (frames, tile_w, tile_h)."""
    h, w = pixels.shape
    if h <= MAX_DIM and w <= MAX_DIM:
        return [pixels], w, h

    tile_w = min(w, MAX_DIM)
    tile_h = min(h, MAX_DIM)
    # Both already multiples of 32 from rewrap_to_32

    flat = pixels.flatten()
    ppt = tile_w * tile_h
    n_tiles = -(-len(flat) // ppt)

    padded = np.full(n_tiles * ppt, flat[-1], dtype=np.uint8)
    padded[:len(flat)] = flat

    tiles = [padded[i * ppt:(i + 1) * ppt].reshape(tile_h, tile_w) for i in range(n_tiles)]
    return tiles, tile_w, tile_h


def get_dequantized(tensor: ReaderTensor) -> npt.NDArray[np.float32]:
    data = dequantize(tensor.data, tensor.tensor_type)
    shape = tuple(int(d) for d in tensor.shape)
    return data.reshape(shape).astype(np.float32)


def _write_frames(frames: list[npt.NDArray[np.uint8]], output_dir: str, layer_id: int) -> None:
    """Write a list of frames as BMPs. Single frame -> file, multiple -> directory."""
    if len(frames) == 1:
        write_bmp(frames[0], os.path.join(output_dir, f"{layer_id:06d}.bmp"))
    else:
        frame_dir = os.path.join(output_dir, f"{layer_id:06d}")
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            write_bmp(frame, os.path.join(frame_dir, f"{i:06d}.bmp"))


def export_tensor(tensor: ReaderTensor, output_dir: str, layer_id: int) -> dict:
    """Export a single tensor as BMP image(s). Returns manifest entry."""
    data = get_dequantized(tensor)
    ndim = data.ndim
    entry = {
        "name": tensor.name,
        "shape": [int(d) for d in tensor.shape],
        "tensor_type": tensor.tensor_type.name,
        "n_elements": int(tensor.n_elements),
        "layer_id": layer_id,
    }

    if ndim <= 2:
        if ndim == 1:
            img_data = wrap_1d(data)
        else:
            img_data = data

        pixels, vmin, vmax = linearize_to_uint8(img_data)
        pixels, rotated = orient_landscape(pixels)
        pixels = rewrap_to_32(pixels)
        pre_h, pre_w = pixels.shape
        frames, tile_w, tile_h = tile_oversized(pixels)
        _write_frames(frames, output_dir, layer_id)

        entry.update(vmin=vmin, vmax=vmax, rotated=rotated,
                     image_width=tile_w, image_height=tile_h,
                     n_frames=len(frames), tiled=len(frames) > 1)
        if len(frames) > 1:
            entry.update(pre_tile_width=pre_w, pre_tile_height=pre_h)
        logger.info(f"  -> {layer_id:06d}{'/' if len(frames) > 1 else '.bmp'}"
                     f"  ({tile_w}x{tile_h}, {len(frames)} frame(s))")

    else:
        rows, cols = data.shape[-2], data.shape[-1]
        slices = data.reshape(-1, rows, cols)
        n_slices = slices.shape[0]
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

        _write_frames(all_frames, output_dir, layer_id)
        h, w = all_frames[0].shape
        entry.update(vmin=vmin_list, vmax=vmax_list, rotated=rotated,
                     image_width=w, image_height=h,
                     n_frames=n_slices, tiled=False)
        logger.info(f"  -> {layer_id:06d}/  ({n_slices} slices, {w}x{h})")

    return entry


# --- Metadata serialization ---

def _to_json(val):
    if isinstance(val, (np.integer, np.bool_)):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def serialize_metadata(reader: GGUFReader) -> tuple[str, list[dict]]:
    """Returns (architecture, metadata_entries)."""
    arch = ""
    entries = []
    for name, field in reader.fields.items():
        if name.startswith('GGUF.'):
            continue
        if not field.types:
            continue

        if name == "general.architecture":
            arch = str(field.contents())
            # GGUFWriter sets this from constructor, skip
            continue

        main_type = field.types[0]
        try:
            if main_type == GGUFValueType.ARRAY:
                elem_type = field.types[-1]
                value = field.contents()
                if isinstance(value, list):
                    value = [_to_json(v) for v in value]
                entries.append({"key": name, "type": f"ARRAY:{elem_type.name}", "value": value})
            else:
                entries.append({"key": name, "type": main_type.name, "value": _to_json(field.contents())})
        except Exception as e:
            logger.warning(f"Skipping metadata '{name}': {e}")

    return arch, entries


# --- Image grouping ---

def build_image_groups(tensor_manifests: list[dict]) -> list[dict]:
    """Group all images by (width, height) for video encoding."""
    groups: dict[tuple[int, int], list[str]] = defaultdict(list)
    for t in tensor_manifests:
        lid = t["layer_id"]
        w, h = t["image_width"], t["image_height"]
        if t["n_frames"] == 1:
            groups[(w, h)].append(f"{lid:06d}.bmp")
        else:
            for s in range(t["n_frames"]):
                groups[(w, h)].append(f"{lid:06d}/{s:06d}.bmp")

    return [{"width": w, "height": h, "images": imgs}
            for (w, h), imgs in sorted(groups.items())]


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

    # Serialize metadata
    arch, metadata = serialize_metadata(reader)
    logger.info(f'* Architecture: {arch}')

    # Export tensors
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

    # Build image groups and write manifest
    image_groups = build_image_groups(tensor_manifests)

    manifest = {
        "architecture": arch,
        "metadata": metadata,
        "tensors": tensor_manifests,
        "image_groups": image_groups,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False)
    logger.info(f'* Written {manifest_path}')

    # Create .compressed dir
    compressed_dir = os.path.join(output_dir, '.compressed')
    os.makedirs(compressed_dir, exist_ok=True)

    logger.info('* Done. Next steps:')
    logger.info(f'  1. cd "{output_dir}" && python -m gguf.scripts.gguf_compress')
    logger.info(f'  2. cd ".compressed" && python -m gguf.scripts.gguf_decompress')


if __name__ == '__main__':
    main()
