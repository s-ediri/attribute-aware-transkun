import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import moduleconf
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from transkun import Data
from transkun.TrainUtil import initializeCheckpoint, save_checkpoint
from transkun.train import current_git_revision, seed_everything


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metadata(args, model_module, model_config):
    files = {}
    for name, path in (
            ("model_config", args.config),
            ("train_metadata", args.metadata),
            ("validation_metadata", args.validation_metadata),
            ("pretrained_checkpoint", args.pretrained_checkpoint)):
        files[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": 1,
        "workflow": "bounded_optimizer_pilot",
        "system_id": args.system_id,
        "seed": args.seed,
        "model_module": model_module,
        "model_config": copy.deepcopy(model_config.__dict__),
        "training": {
            "n_processes": 1,
            "batch_size": 1,
            "hop_size_seconds": args.hop_size,
            "chunk_size_seconds": args.chunk_size,
            "data_loader_workers": 0,
            "max_learning_rate": args.max_lr,
            "weight_decay": args.weight_decay,
            "optimizer_steps": args.schedule_steps,
            "augmentation": False,
            "strict_pretrained": args.strict_pretrained,
            "pedal_extension": False,
            "gradient_clip_norm": args.gradient_clip_norm,
            "deterministic_indexed_samples": True,
        },
        "dataset_root": str(args.dataset_path.resolve()),
        "files": files,
        "git": current_git_revision(),
    }


def validate_resume_metadata(saved, proposed):
    if not saved:
        raise ValueError("Pilot checkpoint has no reproducibility metadata")
    comparisons = {
        "workflow": (saved.get("workflow"), proposed.get("workflow")),
        "system_id": (saved.get("system_id"), proposed.get("system_id")),
        "seed": (saved.get("seed"), proposed.get("seed")),
        "model_module": (saved.get("model_module"), proposed.get("model_module")),
        "model_config": (saved.get("model_config"), proposed.get("model_config")),
        "training": (saved.get("training"), proposed.get("training")),
        "dataset_root": (saved.get("dataset_root"), proposed.get("dataset_root")),
        "files": (saved.get("files"), proposed.get("files")),
    }
    mismatches = [name for name, values in comparisons.items()
                  if values[0] != values[1]]
    if mismatches:
        raise ValueError(
            "Resume arguments do not match pilot checkpoint metadata for: "
            + ", ".join(mismatches))
    return saved


def parameter_checksum(model):
    return float(sum(
        parameter.detach().double().sum().item()
        for parameter in model.parameters()))


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    if not state:
        raise ValueError("Pilot checkpoint has no RNG state for exact resume")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def load_or_initialize(args, model_class, model_config, device, metadata):
    if args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location=device,
                                weights_only=False)
        validate_resume_metadata(checkpoint.get("metadata"), metadata)
        _, _, model, loss_tracker, best_state, optimizer, scheduler = \
            initializeCheckpoint(
                model_class, device, args.max_lr, args.weight_decay,
                args.schedule_steps, model_config)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        restore_rng_state(checkpoint.get("rng_state"))
        return (checkpoint["epoch"], checkpoint["nIter"], model,
                loss_tracker, checkpoint["best_state_dict"], optimizer,
                scheduler, True)

    result = initializeCheckpoint(
        model_class, device, args.max_lr, args.weight_decay,
        args.schedule_steps, model_config,
        pretrainedCheckpoint=args.pretrained_checkpoint,
        strictPretrained=args.strict_pretrained)
    return (*result, False)


def main():
    parser = argparse.ArgumentParser(
        description="Run a bounded real-data optimizer/checkpoint pilot.")
    parser.add_argument("--system-id", choices=("B", "C", "D"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--schedule-steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--chunk-size", type=float, default=2.0)
    parser.add_argument("--hop-size", type=float, default=8.0)
    parser.add_argument("--max-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=40.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strict-pretrained", action="store_true")
    args = parser.parse_args()

    if args.target_steps < 1:
        parser.error("--target-steps must be positive")
    if args.schedule_steps < args.target_steps:
        parser.error("--schedule-steps must be at least --target-steps")
    if args.system_id == "B" and not args.strict_pretrained:
        parser.error("System B requires --strict-pretrained")
    if args.system_id != "B" and args.strict_pretrained:
        parser.error("Only System B uses --strict-pretrained")

    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    configuration = moduleconf.parseFromFile(str(args.config))
    model_class = configuration["Model"].module.TransKun
    model_config = configuration["Model"].config
    model_module = configuration["Model"].module.__name__
    metadata = build_metadata(args, model_module, model_config)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    (epoch, start_step, model, loss_tracker, best_state, optimizer,
     scheduler, resumed) = load_or_initialize(
        args, model_class, model_config, device, metadata)
    if start_step > args.target_steps:
        raise ValueError(
            f"Checkpoint is already at step {start_step}, beyond target "
            f"{args.target_steps}")

    dataset = Data.DatasetMaestro(str(args.dataset_path), str(args.metadata))
    iterator = Data.DatasetMaestroIterator(
        dataset, hopSizeInSecond=args.hop_size,
        chunkSizeInSecond=args.chunk_size,
        notesStrictlyContained=False, ditheringFrames=False, seed=args.seed)

    initial_checksum = parameter_checksum(model)
    losses = []
    learning_rates = []
    step_seconds = []
    note_counts = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    for step in range(start_step, args.target_steps):
        sample = iterator[step % len(iterator)]
        audio = torch.from_numpy(sample["audioSlice"]).unsqueeze(0).to(device)
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        log_probability = model.log_prob(audio, [sample["notes"]])
        raw_loss = -log_probability.sum(-1).mean()
        if not torch.isfinite(raw_loss):
            raise RuntimeError(f"Non-finite loss at step {step + 1}")
        (raw_loss / 50).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip_norm)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"Non-finite gradient norm at step {step + 1}")
        learning_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_seconds.append(time.perf_counter() - started)
        losses.append(raw_loss.item())
        note_counts.append(len(sample["notes"]))
        print(json.dumps({
            "system_id": args.system_id,
            "step": step + 1,
            "loss": raw_loss.item(),
            "gradient_norm": gradient_norm.item(),
            "learning_rate": learning_rates[-1],
            "seconds": step_seconds[-1],
        }))

    final_checksum = parameter_checksum(model)
    loss_tracker["train"].extend(losses)
    save_checkpoint(
        args.checkpoint, epoch, args.target_steps, model, loss_tracker,
        best_state, optimizer, scheduler, metadata=metadata)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved["rng_state"] = capture_rng_state()
    torch.save(saved, args.checkpoint)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved["nIter"] != args.target_steps or saved["metadata"] != metadata:
        raise RuntimeError("Saved pilot checkpoint failed validation")

    result = {
        "protocol": metadata,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "resumed": resumed,
        "start_step": start_step,
        "target_step": args.target_steps,
        "steps_executed": args.target_steps - start_step,
        "losses": losses,
        "learning_rates": learning_rates,
        "step_seconds": step_seconds,
        "note_counts": note_counts,
        "parameter_checksum_before": initial_checksum,
        "parameter_checksum_after": final_checksum,
        "parameter_checksum_changed": initial_checksum != final_checksum,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else None),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved pilot checkpoint to {args.checkpoint}")
    print(f"Saved pilot report to {args.report}")


if __name__ == "__main__":
    main()