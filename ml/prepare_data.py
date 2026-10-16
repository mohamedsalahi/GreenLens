"""Inspect the raw PlantVillage subset and build train/val/test splits.

Reads the extracted dataset directory, verifies every image, removes
exact duplicates (identical file content), and writes two files:

  data/processed/manifest.json   split assignment per image
  data/processed/inventory.json  counts, classes, image properties

The split is stratified per class with a fixed seed. Duplicates are
dropped before splitting so the same picture cannot appear in two splits.

Usage:
  python ml/prepare_data.py --raw data/raw/PlantVillage --out data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from preprocess import load_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_classes(raw_dir: Path) -> list[str]:
    classes = sorted(
        d.name for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not classes:
        raise SystemExit(f"no class directories found under {raw_dir}")
    return classes


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_image(path: Path) -> dict:
    """Basic properties; raises if the file cannot be decoded."""
    with Image.open(path) as img:
        img.verify()
    with Image.open(path) as img:
        arr = load_image(path)
        return {
            "width": img.size[0],
            "height": img.size[1],
            "mode": img.mode,
            "array_shape": list(arr.shape),
            "mean_pixel": float(arr.mean()),
        }


def stratified_split(
    paths_per_class: dict[str, list[Path]],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[Path]]:
    """Split each class independently, so every class keeps its share."""
    rng = random.Random(seed)
    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for cls in sorted(paths_per_class):
        paths = sorted(paths_per_class[cls])
        rng.shuffle(paths)
        n = len(paths)
        n_test = max(1, round(n * test_fraction))
        n_val = max(1, round(n * val_fraction))
        if n < 3:
            raise SystemExit(f"class {cls} has only {n} images, cannot split")
        # Keep train as the remainder so all images are used.
        test = paths[:n_test]
        val = paths[n_test : n_test + n_val]
        train = paths[n_test + n_val :]
        splits["train"].extend(train)
        splits["val"].extend(val)
        splits["test"].extend(test)
    return splits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/PlantVillage"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()

    raw_dir = args.raw
    if not raw_dir.is_dir():
        raise SystemExit(f"dataset directory not found: {raw_dir}")
    args.out.mkdir(parents=True, exist_ok=True)

    classes = find_classes(raw_dir)
    print(f"found {len(classes)} classes under {raw_dir}")

    per_class: dict[str, list[Path]] = defaultdict(list)
    corrupt: list[str] = []
    properties: list[dict] = []
    seen_hash_to_path: dict[str, Path] = {}
    duplicates: list[dict] = []

    for cls in classes:
        files = sorted(
            p
            for p in (raw_dir / cls).iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"  {cls}: {len(files)} files")
        for path in files:
            try:
                props = inspect_image(path)
            except Exception as exc:  # noqa: BLE001 - report and keep going
                corrupt.append(f"{path}: {exc}")
                continue
            digest = file_hash(path)
            if digest in seen_hash_to_path:
                duplicates.append(
                    {"dropped": str(path), "kept": str(seen_hash_to_path[digest])}
                )
                continue
            seen_hash_to_path[digest] = path
            per_class[cls].append(path)
            properties.append(
                {
                    "path": str(path),
                    "class": cls,
                    "sha256": digest,
                    **props,
                }
            )

    kept_total = sum(len(v) for v in per_class.values())
    print(
        f"kept {kept_total} images, "
        f"dropped {len(duplicates)} exact duplicates, "
        f"{len(corrupt)} unreadable files"
    )

    splits = stratified_split(
        per_class, args.val_fraction, args.test_fraction, args.seed
    )

    # Verify the split really is disjoint before writing anything.
    all_paths = splits["train"] + splits["val"] + splits["test"]
    if len(set(all_paths)) != len(all_paths):
        raise SystemExit("split integrity check failed: overlapping files")
    for split, paths in splits.items():
        for cls in classes:
            if not any(p.parent.name == cls for p in paths):
                raise SystemExit(f"split {split} has no images for class {cls}")

    manifest = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "raw_dir": str(raw_dir),
        "classes": classes,
        "splits": {
            split: [{"path": str(p), "class": p.parent.name} for p in paths]
            for split, paths in splits.items()
        },
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    stats = {}
    for split, paths in splits.items():
        stats[split] = dict(Counter(p.parent.name for p in paths))

    inventory = {
        "created": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "classes": classes,
        "class_counts": {cls: len(per_class[cls]) for cls in classes},
        "total_images": kept_total,
        "duplicates_dropped": duplicates,
        "corrupt_files": corrupt,
        "split_counts": stats,
        "image_properties": {
            "unique_sizes": sorted(
                {f"{p['width']}x{p['height']}" for p in properties}
            ),
            "modes": sorted({p["mode"] for p in properties}),
        },
    }
    inventory_path = args.out / "inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=1), encoding="utf-8")

    print(f"\nwrote {manifest_path}")
    print(f"wrote {inventory_path}\n")
    header = f"{'class':38s} {'train':>6s} {'val':>5s} {'test':>5s}"
    print(header)
    print("-" * len(header))
    for cls in classes:
        print(
            f"{cls:38s} {stats['train'].get(cls, 0):6d} "
            f"{stats['val'].get(cls, 0):5d} {stats['test'].get(cls, 0):5d}"
        )
    print(
        f"{'TOTAL':38s} {len(splits['train']):6d} "
        f"{len(splits['val']):5d} {len(splits['test']):5d}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
