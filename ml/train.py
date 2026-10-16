"""Train the VGG16-based leaf disease classifier.

Two-stage transfer learning on the manifest produced by prepare_data.py:

  stage 1  frozen VGG16 backbone, train the classification head
  stage 2  unfreeze the last convolutional block and fine-tune at a low
           learning rate

The best checkpoint by validation accuracy is kept, then evaluated once
on the held-out test split. Artifacts written to --artifacts:

  model.keras           best weights, repacked without the optimizer state
  class_names.json      index to class mapping
  metadata.json         everything needed to reproduce inference
  train_log_phase*.csv  per-epoch history
  test_report.json      held-out metrics and confusion matrix

Usage:
  python ml/train.py --manifest data/processed/manifest.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from preprocess import IMAGE_SIZE, preprocess_path

AUTOTUNE = tf.data.AUTOTUNE


def set_reproducible(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)


def dataset_from_entries(
    entries: list[dict],
    class_to_index: dict[str, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    paths = [e["path"] for e in entries]
    labels = np.asarray([class_to_index[e["class"]] for e in entries], dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path: tf.Tensor, label: tf.Tensor):
        image = tf.numpy_function(preprocess_path, [path], tf.float32)
        image.set_shape((*IMAGE_SIZE, 3))
        return image, label

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 4096), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_model(num_classes: int, dense_units: int, dropout: float) -> tf.keras.Model:
    backbone = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    # Augmentation lives in the model so it runs during training only.
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.05, fill_mode="nearest")(x)
    x = tf.keras.layers.RandomZoom(0.1, fill_mode="nearest")(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(
        dense_units, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inputs, outputs, name="greenlens_vgg16")


def class_weights_from(entries: list[dict], class_to_index: dict[str, int]) -> dict[int, float]:
    counts = Counter(class_to_index[e["class"]] for e in entries)
    total = sum(counts.values())
    present = len(counts)
    return {
        idx: total / (present * count) for idx, count in sorted(counts.items())
    }


def evaluate(
    model: tf.keras.Model,
    ds_test: tf.data.Dataset,
    test_entries: list[dict],
    classes: list[str],
    artifacts_dir: Path,
) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix

    probs = model.predict(ds_test, verbose=0)
    y_pred = probs.argmax(axis=1)
    y_true = np.asarray([classes.index(e["class"]) for e in test_entries])

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    accuracy = float((y_pred == y_true).mean())

    result = {
        "test_accuracy": accuracy,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {
            cls: {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }
            for cls in classes
        },
        "confusion_matrix": matrix.tolist(),
        "classes": classes,
    }
    (artifacts_dir / "test_report.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(matrix, cmap="Greens")
        ax.set_xticks(range(len(classes)), labels=[c[:22] for c in classes], rotation=90)
        ax.set_yticks(range(len(classes)), labels=[c[:22] for c in classes])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(artifacts_dir / "confusion_matrix.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # plotting is optional, metrics are not
        print(f"warning: could not draw confusion matrix: {exc}", file=sys.stderr)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.json"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--phase1-epochs", type=int, default=10)
    parser.add_argument("--phase2-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--dense-units", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision", choices=["mixed_float16", "float32"], default="mixed_float16"
    )
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(f"manifest not found: {args.manifest} (run prepare_data.py first)")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    set_reproducible(args.seed)

    tf.keras.mixed_precision.set_global_policy(args.precision)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    devices = [d.name for d in tf.config.list_physical_devices("GPU")]
    print(f"TensorFlow {tf.__version__}, GPUs: {devices or 'none (CPU)'}")
    print(f"precision policy: {args.precision}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    classes: list[str] = manifest["classes"]
    class_to_index = {c: i for i, c in enumerate(classes)}
    train_entries = manifest["splits"]["train"]
    val_entries = manifest["splits"]["val"]
    test_entries = manifest["splits"]["test"]
    print(
        f"train {len(train_entries)}, val {len(val_entries)}, "
        f"test {len(test_entries)}, classes {len(classes)}"
    )

    ds_train = dataset_from_entries(
        train_entries, class_to_index, args.batch_size, shuffle=True, seed=args.seed
    )
    ds_val = dataset_from_entries(
        val_entries, class_to_index, args.batch_size, shuffle=False, seed=args.seed
    )
    ds_test = dataset_from_entries(
        test_entries, class_to_index, args.batch_size, shuffle=False, seed=args.seed
    )

    model = build_model(len(classes), args.dense_units, args.dropout)
    checkpoint_path = args.artifacts / "model.keras"
    class_weight = None
    if not args.no_class_weights:
        class_weight = class_weights_from(train_entries, class_to_index)
        print("class weights:", {classes[k]: round(v, 2) for k, v in class_weight.items()})

    def callbacks(phase: int):
        return [
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=4,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(args.artifacts / f"train_log_phase{phase}.csv")),
        ]

    print("\n--- stage 1: frozen backbone ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history1 = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.phase1_epochs,
        callbacks=callbacks(1),
        class_weight=class_weight,
    )

    print("\n--- stage 2: fine-tune last conv block ---")
    for layer in model.layers:
        if layer.name.startswith("vgg16"):
            for inner in layer.layers:
                inner.trainable = inner.name.startswith("block5")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.finetune_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.phase2_epochs,
        callbacks=callbacks(2),
        class_weight=class_weight,
    )

    print("\n--- evaluation on held-out test split ---")
    best = tf.keras.models.load_model(checkpoint_path)
    test_result = evaluate(best, ds_test, test_entries, classes, args.artifacts)
    print(f"test accuracy: {test_result['test_accuracy']:.4f}")
    print(f"macro F1:      {test_result['macro_f1']:.4f}")

    # The checkpoint keeps the optimizer state so a run can be resumed, but
    # that more than doubles the file size. Inference only needs the weights,
    # so repack them into a fresh uncompiled model and drop the optimizer.
    compact = build_model(len(classes), args.dense_units, args.dropout)
    compact.set_weights(best.get_weights())
    compact.save(checkpoint_path)

    index_map = {str(i): c for i, c in enumerate(classes)}
    (args.artifacts / "class_names.json").write_text(
        json.dumps({"classes": classes, "index": index_map}, indent=1), encoding="utf-8"
    )

    metadata = {
        "created": datetime.now(timezone.utc).isoformat(),
        "backbone": "VGG16 (ImageNet weights, include_top=False)",
        "weights_source": "keras.applications.VGG16 weights='imagenet'",
        "input_size": list(IMAGE_SIZE),
        "color_order": "RGB input, converted to BGR by preprocess_input",
        "preprocessing": "PIL resize (bilinear) to 224x224, then keras.applications.vgg16.preprocess_input",
        "preprocessing_module": "ml/preprocess.py",
        "classes": classes,
        "num_classes": len(classes),
        "dataset": {
            "manifest": str(args.manifest),
            "train": len(train_entries),
            "val": len(val_entries),
            "test": len(test_entries),
            "seed": manifest["seed"],
        },
        "training": {
            "phase1": {
                "frozen": "entire VGG16 backbone",
                "epochs_ran": len(history1.history["loss"]),
                "optimizer": f"Adam {args.lr}",
            },
            "phase2": {
                "unfrozen": "VGG16 block5",
                "epochs_ran": len(history2.history["loss"]),
                "optimizer": f"Adam {args.finetune_lr}",
            },
            "batch_size": args.batch_size,
            "class_weights": "balanced" if class_weight else "none",
            "precision": args.precision,
            "augmentation": "horizontal flip, rotation 0.05, zoom 0.1 (training only)",
        },
        "test_metrics": {
            "accuracy": test_result["test_accuracy"],
            "macro_f1": test_result["macro_f1"],
            "weighted_f1": test_result["weighted_f1"],
        },
        "environment": {
            "tensorflow": tf.__version__,
            "python": platform.python_version(),
            "gpus": devices,
        },
    }
    (args.artifacts / "metadata.json").write_text(
        json.dumps(metadata, indent=1), encoding="utf-8"
    )
    print(f"artifacts written to {args.artifacts.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
