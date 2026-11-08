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

import numpy as np
import torch

from transkun import Data, Evaluation, ModelTransformer


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


def score_pair(baseline, candidate, references, audio, sample_rate, device):
    baseline_notes = transcribe(baseline, audio, sample_rate, device)
    candidate_notes = transcribe(candidate, audio, sample_rate, device)

    baseline_metrics = Evaluation.compareTranscription(
        baseline_notes,
        references,
        splitPedal=True,
        computeDeviations=False,
        onset_tolerance=0.05,
    )
    candidate_metrics = Evaluation.compareTranscription(
        candidate_notes,
        references,
        splitPedal=True,
        computeDeviations=False,
        onset_tolerance=0.05,
    )

    result = {}
    for metric in METRICS:
        if metric not in baseline_metrics or metric not in candidate_metrics:
            raise KeyError(f"Expected metric {metric!r} was not produced")
        result[metric] = {
            "baseline": [float(value) for value in baseline_metrics[metric]],
            "fine_tuned_10000": [
                float(value) for value in candidate_metrics[metric]
            ],
        }

    return {
        "metrics": result,
        "n_reference": int(baseline_metrics["nGT"]),
        "n_estimated_baseline": int(baseline_metrics["nEst"]),
        "n_estimated_fine_tuned_10000": int(candidate_metrics["nEst"]),
    }


def summarize(records):
    if not records:
        return {}

    deltas = {
        metric: np.asarray(
            [
                record["metrics"][metric]["fine_tuned_10000"][2]
                - record["metrics"][metric]["baseline"][2]
                for record in records
            ],
            dtype=np.float64,
        )
        for metric in METRICS
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(records), size=(BOOTSTRAP_REPLICATES, len(records)))

    summary = {}
    for metric in METRICS:
        baseline_f1 = np.asarray(
            [record["metrics"][metric]["baseline"][2] for record in records],
            dtype=np.float64,
        )
        candidate_f1 = np.asarray(
            [
                record["metrics"][metric]["fine_tuned_10000"][2]
                for record in records
            ],
            dtype=np.float64,
        )
        bootstrap_delta = deltas[metric][indices].mean(axis=1)
        summary[metric] = {
            "baseline_macro_f1": float(baseline_f1.mean()),
            "fine_tuned_10000_macro_f1": float(candidate_f1.mean()),
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the original TransKun checkpoint and the 10k "
            "structured-attribute checkpoint over complete MAESTRO "
            "validation recordings."
        )
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("transkun/pretrained/2.0.conf"))
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=Path("transkun/pretrained/2.0.pt"),
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=Path("checkpoint/structured-attributes-10000.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoint/maestro-baseline-vs-10000-full-validation.json"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for input_path in (
        args.dataset_path,
        args.metadata,
        args.config,
        args.baseline_checkpoint,
        args.candidate_checkpoint,
    ):
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
        "evaluation": "full_piece_paired_validation",
        "dataset_path": str(args.dataset_path.resolve()),
        "metadata_path": str(args.metadata.resolve()),
        "metadata_sha256": sha256_file(args.metadata),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "baseline_architecture_commit": BASELINE_SOURCE_COMMIT,
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "baseline_checkpoint_sha256": sha256_file(args.baseline_checkpoint),
        "candidate_checkpoint": str(args.candidate_checkpoint.resolve()),
        "candidate_checkpoint_sha256": sha256_file(args.candidate_checkpoint),
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
        completed_names = {record["audio_filename"] for record in records}
        if len(completed_names) != len(records):
            raise ValueError("Saved partial results contain duplicate recordings")
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

    baseline_module = load_baseline_module()
    baseline_config = config_from_json(args.config, baseline_module)
    candidate_config = config_from_json(args.config, ModelTransformer)
    baseline = load_model(
        baseline_module,
        baseline_config,
        args.baseline_checkpoint,
        device,
    )
    candidate = load_model(
        ModelTransformer,
        candidate_config,
        args.candidate_checkpoint,
        device,
    )

    start_time = time.time()
    for index in range(piece_count):
        audio_filename = dataset.data[index]["audio_filename"]
        if audio_filename in completed_names:
            continue
        _, references, audio, sample_rate = dataset.getSample(index)
        scored = score_pair(
            baseline,
            candidate,
            references,
            audio,
            sample_rate,
            device,
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
            f"{record['metrics']['note+offset']['fine_tuned_10000'][2] - record['metrics']['note+offset']['baseline'][2]:+.6f}"
        )

    result.update(
        {
            "complete": len(records) == piece_count,
            "completed_pieces": len(records),
            "records": records,
            "summary": summarize(records),
            "elapsed_seconds": time.time() - start_time,
        }
    )
    write_result(args.output, result)
    print(f"Saved {len(records)} piece results to {args.output}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
