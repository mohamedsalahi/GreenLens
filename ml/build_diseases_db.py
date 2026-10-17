#!/usr/bin/env python3
"""Build artifacts/diseases.db from ml/disease_info.json.

The database is keyed by the exact class ids used in the detection JSON
(`artifacts/class_names.json`), so the app can look up guidance for any
prediction straight from its class id.

Tables:
  diseases  one row per class: crop, name, pathogen, feedback, about, ...
  steps     ordered management / treatment instructions per disease
  sources   ordered reference list per disease (extension services,
            research institutions), shown in the app for credibility

Usage:
  python ml/build_diseases_db.py [--info ml/disease_info.json] \
      [--out artifacts/diseases.db]

The .db file is committed so the app runs without a build step; re-run this
script after editing the JSON.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INFO = REPO_ROOT / "ml" / "disease_info.json"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "diseases.db"
DEFAULT_CLASSES = REPO_ROOT / "artifacts" / "class_names.json"

SCHEMA = """
CREATE TABLE diseases (
    class_id TEXT PRIMARY KEY,
    crop     TEXT NOT NULL,
    name     TEXT NOT NULL,
    pathogen TEXT,
    healthy  INTEGER NOT NULL DEFAULT 0,
    feedback TEXT NOT NULL,
    about    TEXT NOT NULL,
    symptoms TEXT,
    spread   TEXT,
    egypt    TEXT
);
CREATE TABLE steps (
    class_id TEXT NOT NULL REFERENCES diseases(class_id),
    section  TEXT NOT NULL CHECK (section IN ('management', 'treatment')),
    position INTEGER NOT NULL,
    step     TEXT NOT NULL,
    PRIMARY KEY (class_id, section, position)
);
CREATE TABLE sources (
    class_id  TEXT NOT NULL REFERENCES diseases(class_id),
    position  INTEGER NOT NULL,
    title     TEXT NOT NULL,
    publisher TEXT NOT NULL,
    url       TEXT NOT NULL,
    PRIMARY KEY (class_id, position)
);
"""


def build(info_path: Path, out_path: Path, classes_path: Path) -> dict[str, int]:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    classes = json.loads(classes_path.read_text(encoding="utf-8"))["classes"]

    missing = [c for c in classes if c not in info]
    extra = [c for c in info if c not in classes]
    if missing or extra:
        raise SystemExit(
            f"disease_info.json does not match class_names.json: "
            f"missing={missing} extra={extra}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)
    connection = sqlite3.connect(out_path)
    try:
        connection.executescript(SCHEMA)
        for class_id in classes:
            entry = info[class_id]
            for field in ("feedback", "about"):
                if not entry.get(field):
                    raise SystemExit(f"{class_id}: {field} must not be empty")
            connection.execute(
                "INSERT INTO diseases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    class_id,
                    entry["crop"],
                    entry["name"],
                    entry.get("pathogen"),
                    int(bool(entry.get("healthy"))),
                    entry["feedback"],
                    entry["about"],
                    entry.get("symptoms"),
                    entry.get("spread"),
                    entry.get("egypt"),
                ),
            )
            for section, key in (("management", "management"), ("treatment", "treatment")):
                for position, step in enumerate(entry.get(key, []), start=1):
                    connection.execute(
                        "INSERT INTO steps VALUES (?, ?, ?, ?)",
                        (class_id, section, position, step),
                    )
            for position, source in enumerate(entry.get("sources", []), start=1):
                if not source["url"].startswith(("http://", "https://")):
                    raise SystemExit(f"{class_id}: bad source url {source['url']!r}")
                connection.execute(
                    "INSERT INTO sources VALUES (?, ?, ?, ?, ?)",
                    (class_id, position, source["title"], source["publisher"], source["url"]),
                )
        connection.commit()

        counts = {
            "diseases": connection.execute("SELECT COUNT(*) FROM diseases").fetchone()[0],
            "steps": connection.execute("SELECT COUNT(*) FROM steps").fetchone()[0],
            "sources": connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
        }
    finally:
        connection.close()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--info", type=Path, default=DEFAULT_INFO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    args = parser.parse_args()

    counts = build(args.info, args.out, args.classes)
    print(
        f"wrote {args.out} with {counts['diseases']} diseases, "
        f"{counts['steps']} instructions and {counts['sources']} sources"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
