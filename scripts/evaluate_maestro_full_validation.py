import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch

from transkun import Data, Evaluation, ModelTransformer, ModelTransformerLegacy


BASELINE_SOURCE_COMMIT = "03b23776bae3c8b3046dcbe1f2b17bea64c5b25f"
METRICS = ("note", "note+offset", "note+velocity", "note+velocity+offset")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261112


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(repository_root=REPOSITORY_ROOT):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root,
        check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository_root, check=True, capture_output=True,
        text=True).stdout.splitlines()
    return {
        "repository_root": str(repository_root.resolve()),
        "commit": commit,
        "dirty": bool(status),
        "status": status,
    }


def load_baseline_module():
    source = subprocess.run(
        [
            "git",
            "show",
            f"{BASELINE_SOURCE_COMMIT}:transkun/ModelTransformer.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module_name = "transkun._historical_model_for_full_validation"
    module = types.ModuleType(module_name)
    module.__package__ = "transkun"
    sys.modules[module_name] = module
    exec(
        compile(
            source,
            f"{BASELINE_SOURCE_COMMIT}:transkun/ModelTransformer.py",
            "exec",
        ),
        module.__dict__,
    )
    return module


def model_module(model_kind):
    if model_kind == "historical":
        return load_baseline_module()
    if model_kind == "legacy":
        return ModelTransformerLegacy
    if model_kind == "structured":
        return ModelTransformer
    raise ValueError(f"Unsupported model kind: {model_kind}")


def config_from_json(config_path, model_class):
    configuration = json.loads(config_path.read_text(encoding="utf-8"))["Model"][
        "config"
    ]
    config = model_class.Config()
    for key, value in configuration.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def checkpoint_state(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint.get("best_state_dict", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError(
            f"No best_state_dict or state_dict found in {checkpoint_path}"
        )
    return state


def load_model(model_class, config, checkpoint_path, device):
    model = model_class.TransKun(conf=config)
    state = checkpoint_state(checkpoint_path)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    del state
    return model


def transcribe(model, audio, sample_rate, device):
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    if sample_rate != model.fs:
        import soxr

        audio = soxr.resample(audio, sample_rate, model.fs)
        sample_rate = model.fs
    if sample_rate != model.fs:
        raise ValueError(
            f"Audio resampling failed: received {sample_rate} Hz, expected {model.fs} Hz"
        )

    audio_tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32)).to(device)
    with torch.inference_mode():
        return model.transcribe(audio_tensor)


def score_pair(
        left, right, references, audio, sample_rate, device,
        left_label="left", right_label="right"):
    left_notes = transcribe(left, audio, sample_rate, device)
    right_notes = transcribe(right, audio, sample_rate, device)

    left_metrics = Evaluation.compareTranscription(
        left_notes,
        references,
        splitPedal=True,
        computeDeviations=False,
        onset_tolerance=0.05,
    )
    right_metrics = Evaluation.compareTranscription(
        right_notes,
        references,
        splitPedal=True,
        computeDeviations=False,
        onset_tolerance=0.05,
    )

    result = {}
    for metric in METRICS:
        if metric not in left_metrics or metric not in right_metrics:
            raise KeyError(f"Expected metric {metric!r} was not produced")
        result[metric] = {
            left_label: [float(value) for value in left_metrics[metric]],
            right_label: [float(value) for value in right_metrics[metric]],
        }

    return {
        "metrics": result,
        "n_reference": int(left_metrics["nGT"]),
        "n_estimated": {
            left_label: int(left_metrics["nEst"]),
            right_label: int(right_metrics["nEst"]),
        },
    }


def summarize(
        records, left_label="left", right_label="right",
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED):
    if not records:
        return {}

    deltas = {
        metric: np.asarray(
            [
                record["metrics"][metric][right_label][2]
                - record["metrics"][metric][left_label][2]
                for record in records
            ],
            dtype=np.float64,
        )
        for metric in METRICS
    }
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0, len(records), size=(bootstrap_replicates, len(records)))

    summary = {}
    for metric in METRICS:
        left_f1 = np.asarray(
            [record["metrics"][metric][left_label][2] for record in records],
            dtype=np.float64,
        )
        right_f1 = np.asarray(
            [
                record["metrics"][metric][right_label][2]
                for record in records
            ],
            dtype=np.float64,
        )
        bootstrap_delta = deltas[metric][indices].mean(axis=1)
        summary[metric] = {
            f"{left_label}_macro_f1": float(left_f1.mean()),
            f"{right_label}_macro_f1": float(right_f1.mean()),
            "paired_delta_macro_f1": float(deltas[metric].mean()),
            "paired_piece_bootstrap_95_percentile_interval": [
                float(value)
                for value in np.percentile(bootstrap_delta, [2.5, 97.5])
            ],
            "pieces_improved": int(np.count_nonzero(deltas[metric] > 0)),
            "pieces_declined": int(np.count_nonzero(deltas[metric] < 0)),
            "pieces_unchanged": int(np.count_nonzero(deltas[metric] == 0)),
        }
    return summary


def write_result(path, result):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def resolve_sides(args):
    compatibility_mode = any((
        args.config is not None,
        args.baseline_checkpoint is not None,
        args.candidate_checkpoint is not None,
    ))
    explicit_mode = any((
        args.left_config is not None,
        args.left_checkpoint is not None,
        args.right_config is not None,
        args.right_checkpoint is not None,
    ))
    if compatibility_mode and explicit_mode:
        raise ValueError(
            "Do not combine legacy --config/--baseline-checkpoint/"
            "--candidate-checkpoint flags with explicit left/right flags")

    if compatibility_mode or not explicit_mode:
        config = args.config or Path("transkun/pretrained/2.0.conf")
        return {
            "left": {
                "label": "baseline",
                "model_kind": "historical",
                "config": config,
                "checkpoint": args.baseline_checkpoint
                or Path("transkun/pretrained/2.0.pt"),
            },
            "right": {
                "label": "fine_tuned_10000",
                "model_kind": "structured",
                "config": config,
                "checkpoint": args.candidate_checkpoint
                or Path("checkpoint/structured-attributes-10000.pt"),
            },
        }

    required = {
        "left_config": args.left_config,
        "left_checkpoint": args.left_checkpoint,
        "right_config": args.right_config,
        "right_checkpoint": args.right_checkpoint,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Explicit paired evaluation requires: " + ", ".join(missing))
    if args.left_label == args.right_label:
        raise ValueError("Left and right labels must be distinct")
    return {
        "left": {
            "label": args.left_label,
            "model_kind": args.left_model,
            "config": args.left_config,
            "checkpoint": args.left_checkpoint,
        },
        "right": {
            "label": args.right_label,
            "model_kind": args.right_model,
            "config": args.right_config,
            "checkpoint": args.right_checkpoint,
        },
    }


def side_protocol(side):
    result = {
        "label": side["label"],
        "model_kind": side["model_kind"],
        "config": str(side["config"].resolve()),
        "config_sha256": sha256_file(side["config"]),
        "checkpoint": str(side["checkpoint"].resolve()),
        "checkpoint_sha256": sha256_file(side["checkpoint"]),
    }
    if side["model_kind"] == "historical":
        result["architecture_commit"] = BASELINE_SOURCE_COMMIT
    return result


def validate_saved_records(records, expected_names, left_label, right_label):
    if not isinstance(records, list):
        raise ValueError("Saved records must be a list")
    names = []
    for index, record in enumerate(records):
        try:
            name = record["audio_filename"]
            metrics = record["metrics"]
            estimated = record["n_estimated"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Saved record {index} does not match the paired schema") from error
        if set(metrics) != set(METRICS):
            raise ValueError(f"Saved record {index} has an unexpected metric set")
        for metric in METRICS:
            if set(metrics[metric]) != {left_label, right_label}:
                raise ValueError(
                    f"Saved record {index} has unexpected labels for {metric}")
        if set(estimated) != {left_label, right_label}:
            raise ValueError(
                f"Saved record {index} has unexpected estimated-note labels")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("Saved partial results contain duplicate recordings")
    if names != list(expected_names[:len(names)]):
        raise ValueError(
            "Saved partial results are not an exact prefix of the requested pieces")
    return set(names)


def evaluation_name(split):
    if split not in ("validation", "test"):
        raise ValueError(f"Unsupported evaluation split: {split}")
    return f"full_piece_paired_{split}"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two TransKun checkpoints over complete paired MAESTRO "
            "recordings."
        )
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--left-model", choices=("historical", "legacy", "structured"), default="structured")
    parser.add_argument("--left-config", type=Path)
    parser.add_argument("--left-checkpoint", type=Path)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-model", choices=("historical", "legacy", "structured"), default="structured")
    parser.add_argument("--right-config", type=Path)
    parser.add_argument("--right-checkpoint", type=Path)
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoint/maestro-baseline-vs-10000-full-validation.json"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    sides = resolve_sides(args)

    input_paths = [args.dataset_path, args.metadata]
    for side in sides.values():
        input_paths.extend((side["config"], side["checkpoint"]))
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Required input does not exist: {input_path}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")

    dataset = Data.DatasetMaestro(str(args.dataset_path), str(args.metadata))
    piece_count = len(dataset.data)
    if args.limit is not None:
        piece_count = min(piece_count, args.limit)

    protocol = {
        "evaluation": evaluation_name(args.split),
        "split": args.split,
        "dataset_path": str(args.dataset_path.resolve()),
        "metadata_path": str(args.metadata.resolve()),
        "metadata_sha256": sha256_file(args.metadata),
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "git": git_provenance(),
        "left": side_protocol(sides["left"]),
        "right": side_protocol(sides["right"]),
        "device": str(device),
        "piece_count": piece_count,
        "limit": args.limit,
        "matching": {
            "onset_tolerance_seconds": 0.05,
            "offset_ratio": 0.2,
            "offset_min_tolerance_seconds": 0.05,
            "pedal_extension": False,
            "aggregation": "macro mean of per-piece F1",
        },
        "transcription": {
            "scope": "complete audio file",
            "segment_size_seconds": 16,
            "segment_hop_seconds": 8,
            "discard_second_half": False,
        },
        "bootstrap": {
            "unit": "piece",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 95%",
        },
    }
    expected_names = [
        dataset.data[index]["audio_filename"] for index in range(piece_count)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(
                f"Output exists; choose a new path or pass --resume: {args.output}"
            )
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if result.get("protocol") != protocol:
            raise ValueError(
                "Cannot resume because the saved evaluation protocol does not "
                "match this invocation"
            )
        if result.get("complete"):
            raise ValueError(f"Evaluation is already complete: {args.output}")
        records = result.get("records", [])
        if result.get("completed_pieces") != len(records):
            raise ValueError("Saved completed_pieces does not match the record count")
        completed_names = validate_saved_records(
            records, expected_names,
            sides["left"]["label"], sides["right"]["label"])
    else:
        if args.resume:
            raise FileNotFoundError(f"No partial result exists to resume: {args.output}")
        records = []
        completed_names = set()
        result = {
            "protocol": protocol,
            "complete": False,
            "completed_pieces": 0,
            "records": records,
            "summary": {},
        }
        write_result(args.output, result)

    loaded_models = {}
    for name, side in sides.items():
        module = model_module(side["model_kind"])
        config = config_from_json(side["config"], module)
        loaded_models[name] = load_model(
            module, config, side["checkpoint"], device)

    start_time = time.time()
    for index in range(piece_count):
        audio_filename = dataset.data[index]["audio_filename"]
        if audio_filename in completed_names:
            continue
        _, references, audio, sample_rate = dataset.getSample(index)
        scored = score_pair(
            loaded_models["left"],
            loaded_models["right"],
            references,
            audio,
            sample_rate,
            device,
            left_label=sides["left"]["label"],
            right_label=sides["right"]["label"],
        )
        record = {"audio_filename": audio_filename, **scored}
        records.append(record)
        completed_names.add(audio_filename)
        result.update(
            {
                "complete": False,
                "completed_pieces": len(records),
                "records": records,
            }
        )
        write_result(args.output, result)
        print(
            f"[{len(records)}/{piece_count}] {audio_filename}: "
            f"note+offset delta "
            f"{record['metrics']['note+offset'][sides['right']['label']][2] - record['metrics']['note+offset'][sides['left']['label']][2]:+.6f}"
        )

    result.update(
        {
            "complete": len(records) == piece_count,
            "completed_pieces": len(records),
            "records": records,
            "summary": summarize(
                records,
                left_label=sides["left"]["label"],
                right_label=sides["right"]["label"],
            ),
            "elapsed_seconds": time.time() - start_time,
        }
    )
    write_result(args.output, result)
    print(f"Saved {len(records)} piece results to {args.output}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
