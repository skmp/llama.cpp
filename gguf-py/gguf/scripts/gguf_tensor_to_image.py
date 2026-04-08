#!/usr/bin/env python3
"""Export GGUF tensor layers as BMP images or raw .bin files."""
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

# Float types: pass through as raw f32 binary (lossless)
FLOAT_TYPES = {
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.BF16,
}


# --- BMP writer (8-bit grayscale) ---

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


# --- Float32 <-> 3 grayscale channels ---

def float32_to_sem(data: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Decompose 2D float32 into 3 grayscale images: (sign, exponent, mantissa_top8)."""
    bits = data.view(np.uint32)
    sign = ((bits >> 31) & 1).astype(np.uint8) * 255
    exp = ((bits >> 23) & 0xFF).astype(np.uint8)
    mantissa = ((bits >> 15) & 0xFF).astype(np.uint8)
    return sign, exp, mantissa


# --- Spatial transforms (on 2D float32) ---

def wrap_1d(data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
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


def orient_landscape(data: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], bool]:
    h, w = data.shape
    if h > w:
        return data.T, True
    return data, False


def rewrap_to_32(data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    h, w = data.shape

    if h < 32 or w < 32:
        flat = data.flatten()
        n = len(flat)
        width = 32
        raw_h = -(-n // width)
        pad_h = -(-raw_h // 32) * 32
        out = np.full(raw_h * width, flat[-1], dtype=np.float32)
        out[:n] = flat
        img = out.reshape(raw_h, width)
        if raw_h < pad_h:
            img = np.vstack([img, np.tile(img[-1:], (pad_h - raw_h, 1))])
        return img

    ph = -(-h // 32) * 32
    pw = -(-w // 32) * 32
    if ph == h and pw == w:
        return data
    out = np.empty((ph, pw), dtype=np.float32)
    out[:h, :w] = data
    if pw > w:
        out[:h, w:] = data[:, -1:]
    if ph > h:
        out[h:] = out[h - 1:h]
    return out


MAX_DIM = 4096


def tile_oversized(data: npt.NDArray[np.float32]) -> tuple[list[npt.NDArray[np.float32]], int, int]:
    h, w = data.shape
    if h <= MAX_DIM and w <= MAX_DIM:
        return [data], w, h

    tile_w = min(w, MAX_DIM)
    tile_h = min(h, MAX_DIM)

    flat = data.flatten()
    ppt = tile_w * tile_h
    n_tiles = -(-len(flat) // ppt)

    padded = np.full(n_tiles * ppt, flat[-1], dtype=np.float32)
    padded[:len(flat)] = flat

    tiles = [padded[i * ppt:(i + 1) * ppt].reshape(tile_h, tile_w) for i in range(n_tiles)]
    return tiles, tile_w, tile_h


def get_dequantized(tensor: ReaderTensor) -> npt.NDArray[np.float32]:
    data = dequantize(tensor.data, tensor.tensor_type)
    np_shape = tuple(int(d) for d in reversed(tensor.shape))
    return data.reshape(np_shape).astype(np.float32)


# --- Frame writing ---

def _write_frames(frames: list[npt.NDArray[np.uint8]], output_dir: str, layer_id: int) -> None:
    if len(frames) == 1:
        write_bmp(frames[0], os.path.join(output_dir, f"{layer_id:06d}.bmp"))
    else:
        frame_dir = os.path.join(output_dir, f"{layer_id:06d}")
        os.makedirs(frame_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            write_bmp(frame, os.path.join(frame_dir, f"{i:06d}.bmp"))


def _f32_tile_to_sem_frames(tile: npt.NDArray[np.float32]) -> list[npt.NDArray[np.uint8]]:
    """Convert one f32 tile to 3 grayscale frames: [sign, exponent, mantissa]."""
    s, e, m = float32_to_sem(tile)
    return [s, e, m]


# --- Export ---

def _process_2d(data_2d: npt.NDArray[np.float32]):
    """Spatial transforms on 2D f32: orient, pad, tile. Returns (tiles, rotated, tiled, tw, th, ptw, pth)."""
    h, w = data_2d.shape
    if h >= 32 and w >= 32:
        data_2d, rotated = orient_landscape(data_2d)
    else:
        rotated = False
    data_2d = rewrap_to_32(data_2d)
    pre_h, pre_w = data_2d.shape
    tiles, tile_w, tile_h = tile_oversized(data_2d)
    tiled = len(tiles) > 1
    return tiles, rotated, tiled, tile_w, tile_h, pre_w, pre_h


def export_tensor(tensor: ReaderTensor, output_dir: str, layer_id: int) -> dict:
    data = get_dequantized(tensor)
    ndim = data.ndim
    entry = {
        "name": tensor.name,
        "shape": [int(d) for d in tensor.shape],
        "tensor_type": tensor.tensor_type.name,
        "n_elements": int(tensor.n_elements),
        "layer_id": layer_id,
    }

    # Float types: raw f32 binary (lossless)
    if tensor.tensor_type in FLOAT_TYPES:
        bin_path = os.path.join(output_dir, f"{layer_id:06d}.bin")
        data.astype(np.float32).tofile(bin_path)
        entry["encoding"] = "bin"
        logger.info(f"  -> {layer_id:06d}.bin  ({data.nbytes} bytes, passthrough)")
        return entry

    # Quantized types: 3 grayscale frames per logical frame (sign, exp, mantissa)
    entry["encoding"] = "sem"

    if ndim <= 2:
        img_data = wrap_1d(data) if ndim == 1 else data
        tiles, rotated, tiled, tw, th, ptw, pth = _process_2d(img_data)

        # Each tile -> 3 grayscale frames
        all_frames = []
        for tile in tiles:
            all_frames.extend(_f32_tile_to_sem_frames(tile))
        _write_frames(all_frames, output_dir, layer_id)

        entry.update(rotated=rotated, image_width=tw, image_height=th,
                     n_frames=len(all_frames), tiled=tiled)
        if tiled:
            entry.update(pre_tile_width=ptw, pre_tile_height=pth)
        n_logical = len(tiles)
        logger.info(f"  -> {layer_id:06d}{'/' if len(all_frames) > 1 else '.bmp'}"
                     f"  ({tw}x{th}, {n_logical} tile(s) x3 = {len(all_frames)} frames)")

    else:
        rows, cols = data.shape[-2], data.shape[-1]
        slices = data.reshape(-1, rows, cols)
        n_slices = slices.shape[0]
        rotated = rows > cols and rows >= 32 and cols >= 32

        all_frames = []
        for i in range(n_slices):
            sl = slices[i]
            if rotated:
                sl = sl.T
            sl = rewrap_to_32(sl)
            all_frames.extend(_f32_tile_to_sem_frames(sl))

        _write_frames(all_frames, output_dir, layer_id)
        h, w = all_frames[0].shape
        entry.update(rotated=rotated, image_width=w, image_height=h,
                     n_frames=len(all_frames), tiled=False)
        logger.info(f"  -> {layer_id:06d}/  ({n_slices} slices x3 = {len(all_frames)} frames, {w}x{h})")

    return entry


# --- Metadata ---

def get_architecture(reader: GGUFReader) -> str:
    for name, field in reader.fields.items():
        if name == "general.architecture":
            return str(field.contents())
    return ""


def save_metadata_gguf(reader: GGUFReader, arch: str, output_path: str) -> None:
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
    """Group images by (width, height, channel). SEM frames are grouped separately by channel."""
    groups: dict[tuple[int, int, str], list[str]] = defaultdict(list)
    for t in tensor_manifests:
        if t.get("encoding") not in ("sem",):
            continue
        lid = t["layer_id"]
        w, h = t["image_width"], t["image_height"]
        n_frames = t["n_frames"]
        # SEM: every 3 frames are (sign, exp, mantissa) for one logical frame
        if n_frames == 3:
            # Single logical frame, 3 files in root
            # Actually stored as dir since n_frames > 1
            pass
        # All SEM frames are in a directory: {lid:06d}/{i:06d}.bmp
        # Frames i%3==0 are sign, i%3==1 are exponent, i%3==2 are mantissa
        channel_names = ['sign', 'exp', 'mantissa']
        for i in range(n_frames):
            ch = channel_names[i % 3]
            if n_frames <= 3:
                path = f"{lid:06d}/{i:06d}.bmp"
            else:
                path = f"{lid:06d}/{i:06d}.bmp"
            groups[(w, h, ch)].append(path)

    return [{"width": w, "height": h, "channel": ch, "images": imgs}
            for (w, h, ch), imgs in sorted(groups.items())]


# --- Main ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Export GGUF tensor layers as BMP images / raw bins")
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
