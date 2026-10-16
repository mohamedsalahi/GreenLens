import json
import shutil
import sys
from pathlib import Path

from PIL import Image

import prepare_data


def make_image(path: Path, color=(120, 180, 90), size=(64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_stratified_split_is_disjoint_and_complete():
    per_class = {
        "Alpha": [Path(f"a{i}.jpg") for i in range(20)],
        "Beta": [Path(f"b{i}.jpg") for i in range(10)],
    }
    splits = prepare_data.stratified_split(per_class, 0.15, 0.15, seed=42)
    everything = splits["train"] + splits["val"] + splits["test"]
    assert len(everything) == len(set(everything)) == 30
    for paths in splits.values():
        assert {p.name[0] for p in paths} == {"a", "b"}


def test_stratified_split_is_reproducible():
    per_class = {"Alpha": [Path(f"a{i}.jpg") for i in range(20)]}
    first = prepare_data.stratified_split(per_class, 0.15, 0.15, seed=7)
    second = prepare_data.stratified_split(per_class, 0.15, 0.15, seed=7)
    assert first == second
    third = prepare_data.stratified_split(per_class, 0.15, 0.15, seed=8)
    assert first != third


def test_prepare_end_to_end(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "raw"
    for cls in ("Alpha", "Beta"):
        for i in range(6):
            make_image(raw / cls / f"{i}.png", color=(i * 30, 100, 50))
    # One byte-identical duplicate and one unreadable file.
    shutil.copyfile(raw / "Alpha/0.png", raw / "Alpha/copy-of-0.png")
    (raw / "Beta/broken.png").write_bytes(b"this is not an image")

    out = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["prepare_data.py", "--raw", str(raw), "--out", str(out)]
    )
    assert prepare_data.main() == 0
    capsys.readouterr()

    manifest = json.loads((out / "manifest.json").read_text())
    inventory = json.loads((out / "inventory.json").read_text())

    assert manifest["classes"] == ["Alpha", "Beta"]
    assert inventory["total_images"] == 11
    assert len(inventory["duplicates_dropped"]) == 1
    assert len(inventory["corrupt_files"]) == 1

    everything = [
        entry["path"]
        for split in manifest["splits"].values()
        for entry in split
    ]
    assert len(everything) == len(set(everything)) == 11
    for split in ("train", "val", "test"):
        assert manifest["splits"][split], f"{split} split is empty"
