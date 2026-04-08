#!/usr/bin/env python3
"""Compare two GGUF files: metadata fields and tensor structure."""
from __future__ import annotations

import sys

import numpy as np

from gguf import GGUFReader, GGUFValueType


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <original.gguf> <reconstructed.gguf>", file=sys.stderr)
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]
    a = GGUFReader(path_a, 'r')
    b = GGUFReader(path_b, 'r')

    print(f"=== File A: {path_a}")
    print(f"    Fields: {len(a.fields)}, Tensors: {len(a.tensors)}")
    print(f"=== File B: {path_b}")
    print(f"    Fields: {len(b.fields)}, Tensors: {len(b.tensors)}")
    print()

    # --- Compare metadata fields ---
    print("=== Metadata ===")
    a_keys = set(a.fields.keys())
    b_keys = set(b.fields.keys())

    missing_in_b = a_keys - b_keys
    extra_in_b = b_keys - a_keys

    if missing_in_b:
        print(f"MISSING in B ({len(missing_in_b)}):")
        for k in sorted(missing_in_b):
            print(f"  - {k}")
    if extra_in_b:
        print(f"EXTRA in B ({len(extra_in_b)}):")
        for k in sorted(extra_in_b):
            print(f"  + {k}")

    # Compare values of shared fields
    shared = sorted(a_keys & b_keys)
    mismatched = 0
    for key in shared:
        fa, fb = a.fields[key], b.fields[key]
        if not fa.types or not fb.types:
            continue
        try:
            va = fa.contents()
            vb = fb.contents()
            # Compare types
            if fa.types[0] != fb.types[0]:
                print(f"  TYPE MISMATCH: {key}: {fa.types[0].name} vs {fb.types[0].name}")
                mismatched += 1
                continue
            # Compare values
            if fa.types[0] == GGUFValueType.ARRAY:
                if isinstance(va, list) and isinstance(vb, list):
                    if len(va) != len(vb):
                        print(f"  ARRAY LEN MISMATCH: {key}: {len(va)} vs {len(vb)}")
                        mismatched += 1
                    elif va != vb:
                        diffs = sum(1 for x, y in zip(va, vb) if x != y)
                        print(f"  ARRAY VALUE MISMATCH: {key}: {diffs}/{len(va)} elements differ")
                        mismatched += 1
            elif va != vb:
                sa = repr(va)[:80]
                sb = repr(vb)[:80]
                print(f"  VALUE MISMATCH: {key}: {sa} vs {sb}")
                mismatched += 1
        except Exception as e:
            print(f"  ERROR comparing {key}: {e}")
            mismatched += 1

    if mismatched == 0 and not missing_in_b and not extra_in_b:
        print("  All metadata matches!")
    else:
        print(f"  {mismatched} mismatched, {len(missing_in_b)} missing, {len(extra_in_b)} extra")
    print()

    # --- Compare tensors ---
    print("=== Tensors ===")
    a_tensors = {t.name: t for t in a.tensors}
    b_tensors = {t.name: t for t in b.tensors}

    missing_t = set(a_tensors.keys()) - set(b_tensors.keys())
    extra_t = set(b_tensors.keys()) - set(a_tensors.keys())

    if missing_t:
        print(f"MISSING in B ({len(missing_t)}):")
        for n in sorted(missing_t):
            print(f"  - {n}")
    if extra_t:
        print(f"EXTRA in B ({len(extra_t)}):")
        for n in sorted(extra_t):
            print(f"  + {n}")

    shared_t = sorted(set(a_tensors.keys()) & set(b_tensors.keys()))
    shape_mismatch = 0
    type_mismatch = 0
    data_mismatch = 0

    for name in shared_t:
        ta, tb = a_tensors[name], b_tensors[name]
        sa = list(ta.shape)
        sb = list(tb.shape)
        if sa != sb:
            print(f"  SHAPE MISMATCH: {name}: {sa} vs {sb}")
            shape_mismatch += 1
        if ta.tensor_type != tb.tensor_type:
            print(f"  TYPE MISMATCH: {name}: {ta.tensor_type.name} vs {tb.tensor_type.name}")
            type_mismatch += 1

    if shape_mismatch == 0 and type_mismatch == 0 and not missing_t and not extra_t:
        print(f"  All {len(shared_t)} tensors match structure!")
    else:
        print(f"  {shape_mismatch} shape mismatches, {type_mismatch} type mismatches, "
              f"{len(missing_t)} missing, {len(extra_t)} extra")
    print()

    # --- Compare tensor DATA (first 5 tensors) ---
    print("=== Tensor Data (first 5) ===")
    from gguf import dequantize
    for name in shared_t[:5]:
        ta, tb = a_tensors[name], b_tensors[name]
        try:
            da = dequantize(ta.data, ta.tensor_type).astype(np.float32).flatten()
            db = dequantize(tb.data, tb.tensor_type).astype(np.float32).flatten()
            n = min(len(da), len(db))
            da, db = da[:n], db[:n]
            if n == 0:
                print(f"  {name}: EMPTY")
                continue
            diff = np.abs(da - db)
            max_diff = float(diff.max())
            mean_diff = float(diff.mean())
            # Check if data is just transposed (compare sorted values)
            corr = np.corrcoef(da[:1000], db[:1000])[0, 1] if n >= 1000 else float('nan')
            exact = np.array_equal(da, db)
            print(f"  {name}: exact={exact}, max_diff={max_diff:.6g}, mean_diff={mean_diff:.6g}, "
                  f"corr={corr:.6f}, a[:5]={da[:5]}, b[:5]={db[:5]}")
            if not exact:
                data_mismatch += 1
        except Exception as e:
            print(f"  {name}: ERROR: {e}")
            data_mismatch += 1

    if data_mismatch == 0:
        print(f"  Data matches for checked tensors!")
    else:
        print(f"  {data_mismatch} data mismatches")


if __name__ == '__main__':
    main()
