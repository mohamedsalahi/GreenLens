"""Persistent inference worker for GreenLens.

Loads the trained model once, then serves one JSON request per line on
stdin and writes one JSON response per line on stdout. Diagnostics go to
stderr, never to stdout, so the protocol stays clean.

Request types:

  {"id": 1, "type": "ping"}
  {"id": 2, "type": "predict", "path": "/abs/path/leaf.jpg", "top_k": 3}
  {"id": 3, "type": "shutdown"}

Responses:

  {"type": "ready", ...}                      once, on startup
  {"id": 1, "type": "pong"}
  {"id": 2, "type": "result", "predictions": [...]}
  {"id": 2, "type": "error", "code": "...", "message": "..."}
  {"type": "bye"}                             after a shutdown request

Error codes: file_not_found, invalid_image, inference_failed, bad_request,
unknown_type, startup_failed.

Usage:
  python ml/inference_worker.py --model artifacts/model.keras \
      --classes artifacts/class_names.json --metadata artifacts/metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "artifacts" / "model.keras"
DEFAULT_CLASSES = REPO_ROOT / "artifacts" / "class_names.json"
DEFAULT_METADATA = REPO_ROOT / "artifacts" / "metadata.json"

log = logging.getLogger("greenlens.worker")


def human_label(class_id: str) -> str:
    """Turn a PlantVillage folder name into something readable.

    Potato___Late_blight -> Potato: Late blight
    Pepper__bell___healthy -> Pepper bell: healthy
    Tomato__Tomato_YellowLeaf__Curl_Virus -> Tomato: YellowLeaf Curl Virus
    """
    words = [w for w in class_id.split("_") if w]
    crop = words[0]
    if len(words) > 1 and words[1] == "bell":
        # PlantVillage writes the bell pepper crop as Pepper__bell.
        crop += " bell"
        del words[1]
    disease = words[1:]
    if disease and disease[0].lower() == crop.split()[0].lower():
        # Some tomato folders repeat the crop in the disease name.
        disease = disease[1:]
    return f"{crop}: {' '.join(disease)}" if disease else crop


class Engine:
    """Loads the model and metadata once, predicts many times."""

    def __init__(self, model_path: Path, classes_path: Path, metadata_path: Path):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        from preprocess import load_image, preprocess_array

        self._tf = tf
        self._load_image = load_image
        self._preprocess_array = preprocess_array

        if not model_path.is_file():
            raise FileNotFoundError(f"model file not found: {model_path}")
        if not classes_path.is_file():
            raise FileNotFoundError(f"class mapping not found: {classes_path}")

        self.classes_info = json.loads(classes_path.read_text(encoding="utf-8"))
        self.classes: list[str] = self.classes_info["classes"]
        self.metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )

        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

        log.info("loading model from %s", model_path)
        self.model = tf.keras.models.load_model(model_path)
        out_dim = int(self.model.output_shape[-1])
        if out_dim != len(self.classes):
            raise ValueError(
                f"model outputs {out_dim} classes but class mapping has {len(self.classes)}"
            )
        self.input_size = tuple(self.metadata.get("input_size", (224, 224)))
        self.devices = [d.name for d in tf.config.list_physical_devices("GPU")]
        log.info("model loaded, classes=%d, gpus=%s", len(self.classes), self.devices or "none")

    def ready_message(self) -> dict[str, Any]:
        return {
            "type": "ready",
            "model": str(DEFAULT_MODEL.name),
            "num_classes": len(self.classes),
            "classes": self.classes,
            "labels": [human_label(c) for c in self.classes],
            "input_size": list(self.input_size),
            "preprocessing": self.metadata.get("preprocessing", "unknown"),
            "gpus": self.devices,
        }

    def predict(self, path: str, top_k: int) -> dict[str, Any]:
        import numpy as np

        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(path)

        image = self._load_image(file_path)
        batch = np.expand_dims(self._preprocess_array(image), axis=0)
        probs = self.model(batch, training=False).numpy()[0]

        k = max(1, min(int(top_k), len(self.classes)))
        order = np.argsort(probs)[::-1][:k]
        predictions = [
            {
                "index": int(i),
                "class": self.classes[int(i)],
                "label": human_label(self.classes[int(i)]),
                "probability": float(probs[int(i)]),
            }
            for i in order
        ]

        from PIL import Image

        with Image.open(file_path) as img:
            width, height = img.size

        return {
            "type": "result",
            "path": str(path),
            "original_size": {"width": width, "height": height},
            "predictions": predictions,
        }


def _respond(message: dict[str, Any]) -> None:
    print(json.dumps(message), flush=True)


def _error(request_id: Any, code: str, message: str) -> dict[str, Any]:
    return {"id": request_id, "type": "error", "code": code, "message": message}


def serve(engine: Engine) -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request_id: Any = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            request_id = request.get("id")
            kind = request.get("type")
            if not isinstance(kind, str):
                raise ValueError("request has no type")
        except ValueError as exc:
            _respond(_error(request_id, "bad_request", str(exc)))
            continue

        try:
            if kind == "ping":
                response: dict[str, Any] = {"id": request_id, "type": "pong"}
            elif kind == "predict":
                path = request.get("path")
                if not isinstance(path, str) or not path:
                    _respond(_error(request_id, "bad_request", "predict needs a path"))
                    continue
                top_k = request.get("top_k", 3)
                if not isinstance(top_k, int) or isinstance(top_k, bool):
                    _respond(_error(request_id, "bad_request", "top_k must be an integer"))
                    continue
                response = engine.predict(path, top_k)
            elif kind == "shutdown":
                _respond({"id": request_id, "type": "bye"})
                return 0
            else:
                _respond(_error(request_id, "unknown_type", f"unknown type: {kind!r}"))
                continue
        except FileNotFoundError:
            _respond(_error(request_id, "file_not_found", f"no such file: {path if kind == 'predict' else '?'}"))
            continue
        except Exception as exc:  # noqa: BLE001 - every failure becomes a response
            code = "invalid_image" if kind == "predict" and isinstance(
                exc, (OSError, ValueError)
            ) else "inference_failed"
            _respond(_error(request_id, code, f"{type(exc).__name__}: {exc}"))
            continue

        response["id"] = request_id
        _respond(response)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        engine = Engine(args.model, args.classes, args.metadata)
    except Exception as exc:  # noqa: BLE001 - startup failures are reported on stdout
        _respond(
            {
                "type": "error",
                "code": "startup_failed",
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
        return 1

    _respond(engine.ready_message())
    try:
        return serve(engine)
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
