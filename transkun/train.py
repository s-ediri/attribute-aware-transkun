import os
import sys
import random

# from .Model_ablation import *
# import torch_optimizer as optim

from torch.utils.tensorboard import SummaryWriter

from . import Data
import copy
import time
import numpy as np
import math
import torch.distributed as dist
import torch.multiprocessing as mp

from .TrainUtil import *
import argparse

import moduleconf
import hashlib
import pathlib
import subprocess


def seed_everything(seed, worker_id=0):
    """Seed one training process deterministically from a declared run seed."""
    process_seed = int(seed) + int(worker_id)
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    return process_seed


def sha256_file(filename):
    digest = hashlib.sha256()
    with open(filename, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_revision():
    repository_root = pathlib.Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
                ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(
                ["git", "-C", str(repository_root), "status", "--porcelain"],
                check=True, capture_output=True, text=True).stdout.strip())
        return {"commit": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def effective_data_loader_workers(requested, platform=None):
    """Return the explicitly requested worker count on every platform."""
    requested = int(requested)
    if requested < 0:
        raise ValueError("DataLoader worker count cannot be negative")
    return requested


def scheduler_warmup_complete(global_step, warmup_steps=500):
    """Return whether the run-level scheduler warmup has completed."""
    return global_step > warmup_steps


def periodic_checkpoint_due(completed_steps, interval_steps=500):
    """Return whether a durable checkpoint is due after an optimizer step."""
    return completed_steps > 0 and completed_steps % interval_steps == 0


def remaining_epoch_indices(dataset_length, batch_size, next_batch):
    """Return deterministic sample indices beginning at the saved batch cursor."""
    usable_samples = dataset_length - (dataset_length % batch_size)
    first_sample = next_batch * batch_size
    if first_sample > usable_samples:
        raise ValueError("Checkpoint batch cursor exceeds the epoch length")
    return range(first_sample, usable_samples)


def capture_rng_state():
    """Capture process RNG streams required for an exact single-process resume."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    """Restore process RNG streams from a controlled-training checkpoint."""
    if not state:
        raise ValueError("Checkpoint has no RNG state for exact resume")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def build_training_state(epoch, next_batch, grad_norm_history, epoch_losses):
    """Build the mutable state needed to continue at the exact next batch."""
    return {
        "schema_version": 1,
        "epoch": int(epoch),
        "next_batch": int(next_batch),
        "grad_norm_history": grad_norm_history.state_dict(),
        "epoch_losses": list(epoch_losses),
        "rng_state": capture_rng_state(),
    }


def restore_training_state(state, start_epoch):
    """Restore mutable training state and return cursor, clip history, and losses."""
    if not state:
        raise ValueError(
                "Checkpoint predates exact-resume support and cannot be resumed "
                "by the controlled experiment workflow")
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported training-state schema")
    if state.get("epoch") != start_epoch:
        raise ValueError("Checkpoint epoch and training-state epoch do not match")
    grad_norm_history = MovingBuffer()
    grad_norm_history.load_state_dict(state["grad_norm_history"])
    restore_rng_state(state["rng_state"])
    return state["next_batch"], grad_norm_history, list(state["epoch_losses"])


def checkpoint_resume_epoch(current_epoch, epoch_complete):
    """Return the zero-based epoch where training should resume.

    Periodic checkpoints are written from inside an epoch, so they must retain
    that epoch's data seed. Checkpoints written after validation may advance to
    the next epoch because the current data iterator has been exhausted.
    """
    return current_epoch + 1 if epoch_complete else current_epoch


def build_run_metadata(args, model_module, model_config, seed):
    """Build the immutable run description stored in every new checkpoint."""
    paths = {
        "model_config": args.modelConf,
        "train_metadata": args.datasetMetaFile_train,
        "validation_metadata": args.datasetMetaFile_val,
        "pretrained_checkpoint": args.pretrainedCheckpoint,
    }
    files = {}
    for name, filename in paths.items():
        if filename is not None:
            files[name] = {
                "path": str(pathlib.Path(filename).resolve()),
                "sha256": sha256_file(filename),
            }

    return {
        "schema_version": 1,
        "system_id": args.systemId,
        "seed": int(seed),
        "model_module": model_module,
        "model_config": copy.deepcopy(model_config.__dict__),
        "training": {
            "n_processes": args.nProcess,
            "batch_size": args.batchSize,
            "hop_size_seconds": args.hopSize,
            "chunk_size_seconds": args.chunkSize,
            "data_loader_workers": args.dataLoaderWorkers,
            "max_learning_rate": args.max_lr,
            "weight_decay": args.weight_decay,
            "optimizer_steps": args.nIter,
            "augmentation": bool(args.augment),
            "strict_pretrained": bool(args.strictPretrained),
            "pedal_extension": bool(args.pedalExtension),
            "deterministic_data_order": True,
            "checkpoint_interval_steps": 500,
        },
        "dataset_root": str(pathlib.Path(args.datasetPath).resolve()),
        "files": files,
        "git": current_git_revision(),
        "runtime": {
            "platform": sys.platform,
            "effective_data_loader_workers": effective_data_loader_workers(
                    args.dataLoaderWorkers),
        },
    }


def resolve_run_metadata(filename, proposed):
    """Preserve initialization provenance and reject incompatible resumes."""
    if not os.path.exists(filename):
        return proposed

    checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
    saved = checkpoint.get("metadata", {})
    if not saved:
        raise ValueError(
                "Existing checkpoint has no reproducibility metadata and "
                "cannot be resumed by the controlled experiment workflow: "
                f"{filename}")

    saved_training = dict(saved.get("training", {}))
    proposed_training = dict(proposed.get("training", {}))
    # Initialization strictness is provenance for a new run, not a resume
    # setting. A resume intentionally omits --pretrainedCheckpoint and
    # --strictPretrained because it restores the saved optimizer/model state.
    saved_training.pop("strict_pretrained", None)
    proposed_training.pop("strict_pretrained", None)

    comparisons = {
        "system_id": (saved.get("system_id"), proposed.get("system_id")),
        "seed": (saved.get("seed"), proposed.get("seed")),
        "model_module": (saved.get("model_module"), proposed.get("model_module")),
        "model_config": (saved.get("model_config"), proposed.get("model_config")),
        "training": (saved_training, proposed_training),
        "dataset_root": (saved.get("dataset_root"), proposed.get("dataset_root")),
        "model_config_file": (
            saved.get("files", {}).get("model_config"),
            proposed.get("files", {}).get("model_config")),
        "train_metadata_file": (
            saved.get("files", {}).get("train_metadata"),
            proposed.get("files", {}).get("train_metadata")),
        "validation_metadata_file": (
            saved.get("files", {}).get("validation_metadata"),
            proposed.get("files", {}).get("validation_metadata")),
    }
    mismatches = [
        name for name, (old, new) in comparisons.items() if old != new]
    if mismatches:
        raise ValueError(
                "Resume arguments do not match checkpoint metadata for: "
                + ", ".join(mismatches))

    resolved = copy.deepcopy(saved)
    if proposed.get("runtime"):
        resolved["runtime"] = copy.deepcopy(proposed["runtime"])
    resume_git = proposed.get("git")
    if resume_git and resume_git != saved.get("git"):
        history = resolved.setdefault("resume_git_history", [])
        entry = {
            "start_step": checkpoint.get("nIter"),
            "git": copy.deepcopy(resume_git),
        }
        if not history or history[-1] != entry:
            history.append(entry)
    return resolved


def train(workerId, nWorker, filename, runSeed, args):
    if nWorker != 1:
        raise ValueError(
                "Exact controlled training currently requires --nProcess 1")
    if args.pretrainedCheckpoint is not None:
        if os.path.exists(filename):
            raise FileExistsError(
                    "Cannot initialize from pretrained weights because the "
                    f"training checkpoint already exists: {filename}")
        if not os.path.isfile(args.pretrainedCheckpoint):
            raise FileNotFoundError(
                    "Pretrained checkpoint not found: "
                    f"{args.pretrainedCheckpoint}")

    parallel = True
    if nWorker == 1:
        parallel = False

    if parallel:
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        dist.init_process_group('nccl', rank=workerId, world_size=nWorker)

    device = torch.device("cuda:"+str(workerId%torch.cuda.device_count()) if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    processSeed = seed_everything(runSeed, workerId)
    # torch.autograd.set_detect_anomaly(True)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True 
        torch.backends.cudnn.allow_tf32 = True 


    # obtain the Model Module
    confManager = moduleconf.parseFromFile(args.modelConf)
    TransKun = confManager["Model"].module.TransKun
    conf = confManager["Model"].config
    modelModule = confManager["Model"].module.__name__
    if args.hopSize is None:
        args.hopSize = conf.segmentHopSizeInSecond
    if args.chunkSize is None:
        args.chunkSize = conf.segmentSizeInSecond
    runMetadata = build_run_metadata(args, modelModule, conf, runSeed)
    runMetadata = resolve_run_metadata(filename, runMetadata)



    if workerId == 0:
        print("Run seed: {} (worker seed: {})".format(runSeed, processSeed))
        # if the saved file does not exist
        if not os.path.exists(filename):
            print("initializing the model...")

            startEpoch, startIter, model,  lossTracker, best_state_dict, optimizer, lrScheduler = initializeCheckpoint(
                                TransKun,
                                device = device,
                                max_lr= args.max_lr,
                                weight_decay = args.weight_decay,
                                nIter = args.nIter,
                                conf = conf,
                                pretrainedCheckpoint=args.pretrainedCheckpoint,
                                strictPretrained=args.strictPretrained)


            save_checkpoint(
                    filename, startEpoch, startIter, model, lossTracker,
                    best_state_dict, optimizer, lrScheduler,
                    metadata=runMetadata,
                    training_state=build_training_state(
                            startEpoch, 0, MovingBuffer(initValue=40, maxLen=10000),
                            []))

    if parallel:
        dist.barrier()
        

    (startEpoch, startIter, model, lossTracker, best_state_dict, optimizer,
     lrScheduler, savedTrainingState) = load_checkpoint(
             TransKun, conf, filename, device, return_training_state=True)
    print(
            "Worker {} loaded checkpoint at step {}/{}.".format(
                    workerId, startIter, args.nIter))
    if startIter >= args.nIter:
        if workerId == 0:
            print(f"Training already reached the requested {args.nIter} steps.")
        if parallel:
            dist.destroy_process_group()
        return

    startBatch, gradNormHist, resumedEpochLosses = restore_training_state(
            savedTrainingState, startEpoch)

    if workerId == 0:
        print("Loading training and validation datasets...")

    datasetPath = args.datasetPath
    datasetPicklePath = args.datasetMetaFile_train
    datasetPicklePath_val = args.datasetMetaFile_val

    dataset = Data.DatasetMaestro(datasetPath, datasetPicklePath)
    datasetVal = Data.DatasetMaestro(datasetPath, datasetPicklePath_val)

    print("Worker {} loaded datasets.".format(workerId))


    if workerId == 0:
        writer = SummaryWriter(filename+".log", purge_step=startIter)

    globalStep = startIter
    # create dataloader


    # this iterator should be constructed each time
    batchSize = args.batchSize

    hopSize = args.hopSize
    chunkSize = args.chunkSize

    augmentator = None
    if args.augment:
        augmentator = Data.AugmentatorAudiomentations(sampleRate=44100, noiseFolder = args.noiseFolder, convIRFolder = args.irFolder)

    dataLoaderWorkers = effective_data_loader_workers(args.dataLoaderWorkers)
    dataLoaderOptions = {"num_workers": dataLoaderWorkers}
    if dataLoaderWorkers > 0:
        dataLoaderOptions["prefetch_factor"] = max(4, dataLoaderWorkers)

    trainingComplete = False
    for epoc in range(startEpoch, 1000000):
        # average length will be chunkSize
        dataIter = Data.DatasetMaestroIterator(dataset, hopSize, chunkSize, seed = epoc*100+runSeed, augmentator = augmentator, notesStrictlyContained=False)

        epochStartBatch = startBatch if epoc == startEpoch else 0
        totalEpochBatches = len(dataIter) // batchSize
        remainingIndices = remaining_epoch_indices(
                len(dataIter), batchSize, epochStartBatch)
        remainingData = torch.utils.data.Subset(dataIter, remainingIndices)
        loaderGenerator = torch.Generator()
        loaderGenerator.manual_seed(runSeed + epoc * 100 + 1)
        dataloader = torch.utils.data.DataLoader(
                remainingData, batch_size=batchSize,
                collate_fn=Data.collate_fn_batching, shuffle=False,
                drop_last=True, generator=loaderGenerator, **dataLoaderOptions)


        lossAll = list(resumedEpochLosses) if epoc == startEpoch else []

        # curLRScheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 5e-5, max_lr = 1e-4, step_size_up = len(dataloader)//10*5, step_size_down = len(dataloader)//10*5, cycle_momentum=False)
        for idx, batch in enumerate(dataloader):
            absoluteBatch = epochStartBatch + idx
            if workerId ==0:
                currentLR = [p["lr"] for p in optimizer.param_groups][0]
                writer.add_scalar(f'Optimizer/lr', currentLR , globalStep)

            computeStats = False 
            if absoluteBatch % 40 == 0:
                computeStats =True


            t1 = time.time()

            model.train()
            optimizer.zero_grad()

            totalBatch= torch.zeros(1).cuda()
            totalLoss = torch.zeros(1).cuda()
            totalLen = torch.zeros(1).cuda()


            totalGT= torch.zeros(1).cuda()
            totalEst = torch.zeros(1).cuda()
            totalCorrect = torch.zeros(1).cuda()

            totalGTFramewise = torch.zeros(1).cuda()
            totalEstFramewise = torch.zeros(1).cuda()
            totalCorrectFramewise = torch.zeros(1).cuda()

            totalSEVelocity = torch.zeros(1).cuda()
            totalSEOF= torch.zeros(1).cuda()



            # notesBatch = [sample["notes"] for sample in batch]
            # audioSlices = [torch.from_numpy(sample["audioSlice"]) for sample in batch]
            # # make sure all having the same number of samples

            # nAudioSamplesMin = min( [_.shape[0] for _ in audioSlices])
            # nAudioSamplesMax = max( [_.shape[0] for _ in audioSlices])

            # assert nAudioSamplesMax-nAudioSamplesMin < 2
            
            # audioSlices = [_[:nAudioSamplesMin] for _ in audioSlices]

            # audioSlices = torch.stack(
                    # audioSlices, dim = 0) . to(device)

            notesBatch = batch["notes"]
            audioSlices = batch["audioSlices"].to(device)

            audioLength = audioSlices.shape[1]/model.conf.fs

            logp = model.log_prob(audioSlices, notesBatch)
            loss = (-logp.sum(-1).mean())

            (loss/50).backward()

            totalBatch = totalBatch + 1
            totalLen = totalLen + audioLength
            totalLoss = totalLoss + loss.detach()


            if computeStats:
                with torch.no_grad():
                    model.eval()
                    stats = model.computeStats(audioSlices, notesBatch)
                    stats2 = model.computeStatsMIREVAL(audioSlices, notesBatch)
                    totalGT = totalGT+ stats2["nGT"]
                    totalEst= totalEst + stats2["nEst"]
                    totalCorrect= totalCorrect+ stats2["nCorrect"]
                    totalGTFramewise = totalGTFramewise+ stats["nGTFramewise"]
                    totalEstFramewise = totalEstFramewise + stats["nEstFramewise"]
                    totalCorrectFramewise = totalCorrectFramewise+ stats["nCorrectFramewise"]
                    totalSEVelocity = totalSEVelocity + stats["seVelocityForced"]
                    totalSEOF = totalSEOF + stats["seOFForced"]


            # checkNoneGradient(model)
    
            if parallel:
                dist.all_reduce(totalLoss.data)
                dist.all_reduce(totalLen.data)
                dist.all_reduce(totalBatch.data)
                if computeStats:
                    dist.all_reduce(totalGT.data)
                    dist.all_reduce(totalEst.data)
                    dist.all_reduce(totalCorrect.data)
                    dist.all_reduce(totalGTFramewise.data)
                    dist.all_reduce(totalEstFramewise.data)
                    dist.all_reduce(totalCorrectFramewise.data)
                    dist.all_reduce(totalSEOF.data)
                    dist.all_reduce(totalSEVelocity.data)


                average_gradients(model, totalLen, parallel)

            # compute gradient clipping value

            # communicate gradient clipping values


            loss = totalLoss/totalLen

            # adaptive gradient clipping
            curClipValue = gradNormHist.getQuantile(args.gradClippingQuantile)


            totalNorm = torch.nn.utils.clip_grad_norm_(model.parameters(), curClipValue)

            gradNormHist.step(totalNorm.item())

            optimizer.step()
            # curLRScheduler.step()
            
            try:
                if scheduler_warmup_complete(globalStep):
                    lrScheduler.step()
            except:
                # continue after the final iteration
                pass


            if workerId == 0:
                t2 = time.time()
                print(
                        "epoch:{} progress:{:.1f}% step:{}/{} "
                        "lossPerSecond:{:.4f} learningRate:{:.2e} "
                        "gradNorm:{:.2f} clipThreshold:{:.2f} batchTime:{:.2f}s".format(
                                epoc + 1,
                                100 * (absoluteBatch + 1) / totalEpochBatches,
                                globalStep + 1, args.nIter, loss.item(),
                                currentLR, totalNorm.item(), curClipValue, t2 - t1))
                writer.add_scalar(f'Loss/train', loss.item(), globalStep)
                writer.add_scalar(f'Optimizer/gradNorm', totalNorm.item(), globalStep)
                writer.add_scalar(f'Optimizer/clipValue', curClipValue, globalStep)
                if computeStats:
                    nGTCount = totalGT.item()
                    nEstCount = totalEst.item()
                    nCorrectCount = totalCorrect.item()
                    nGT = nGTCount + 1e-4
                    nEst = nEstCount + 1e-4
                    nCorrect = nCorrectCount + 1e-4
                    precision = nCorrect/nEst
                    recall = nCorrect/nGT
                    f1 = 2* precision*recall/(precision+recall)              
                    print(
                            "Note+offset counts: groundTruth={:.1f} "
                            "estimated={:.1f} matched={:.1f}".format(
                                    nGTCount, nEstCount, nCorrectCount))
                    
                    
                    writer.add_scalar(f'Loss/train_f1', f1, globalStep)
                    writer.add_scalar(f'Loss/train_precision', precision, globalStep)
                    writer.add_scalar(f'Loss/train_recall', recall, globalStep)

                    nGTFramewise = totalGTFramewise.item()+1e-4
                    nEstFramewise = totalEstFramewise.item()+1e-4
                    nCorrectFramewise = totalCorrectFramewise.item()+1e-4
                    precisionFrame = nCorrectFramewise/nEstFramewise
                    recallFrame = nCorrectFramewise/nGTFramewise
                    f1Frame = 2* precisionFrame*recallFrame/(precisionFrame+recallFrame)              

                    mseVelocity = totalSEVelocity.item()/nGT
                    mseOF = totalSEOF.item()/nGT
                    
                    writer.add_scalar(f'Loss/train_f1_frame', f1Frame, globalStep)
                    writer.add_scalar(f'Loss/train_precision_frame', precisionFrame, globalStep)
                    writer.add_scalar(f'Loss/train_recall_frame', recallFrame, globalStep)
                    writer.add_scalar(f'Loss/train_mse_velocity', mseVelocity, globalStep)
                    writer.add_scalar(f'Loss/train_mse_OF', mseOF, globalStep)
                    print(
                            "Note+offset: F1={:.4f} precision={:.4f} "
                            "recall={:.4f}".format(f1, precision, recall))
                    print(
                            "Framewise: F1={:.4f} precision={:.4f} "
                            "recall={:.4f}".format(
                                    f1Frame, precisionFrame, recallFrame))
                    print(
                            "Attribute MSE: velocity={:.4f} "
                            "onsetOffsetTiming={:.4f}".format(mseVelocity, mseOF))




                if math.isnan(loss.item()):
                    exit()
                lossAll.append(loss.item())


                completedSteps = globalStep + 1
                nextBatch = absoluteBatch + 1
                if (completedSteps < args.nIter
                        and periodic_checkpoint_due(completedSteps)):
                    save_checkpoint(
                            filename,
                            checkpoint_resume_epoch(epoc, epoch_complete=False),
                            completedSteps, model, lossTracker,
                            best_state_dict, optimizer, lrScheduler,
                            metadata=runMetadata,
                            training_state=build_training_state(
                                    epoc, nextBatch, gradNormHist, lossAll))
                    print(
                            "Checkpoint saved at step {}/{}: {}".format(
                                    completedSteps, args.nIter, filename))

            globalStep+= 1
            if globalStep >= args.nIter:
                trainingComplete = True
                break
            # torch.cuda.empty_cache()


        if workerId == 0:
            print(
                    "Validating after epoch {} at step {}/{}...".format(
                            epoc + 1, globalStep, args.nIter))
        # let's do validation
        torch.cuda.empty_cache()
        

        
        dataIterVal = Data.DatasetMaestroIterator(datasetVal, hopSizeInSecond = conf.segmentHopSizeInSecond, chunkSizeInSecond = chunkSize, notesStrictlyContained=False, seed = runSeed+epoc*100) 
        if parallel:
            samplerVal = torch.utils.data.distributed.DistributedSampler(dataIterVal)
            dataloaderVal= torch.utils.data.DataLoader(dataIterVal, batch_size = 2*batchSize, collate_fn = Data.collate_fn , num_workers=dataLoaderWorkers, sampler = samplerVal)
        else:
            dataloaderVal= torch.utils.data.DataLoader(
                    dataIterVal, batch_size=2*batchSize,
                    collate_fn=Data.collate_fn, num_workers=dataLoaderWorkers,
                    shuffle=False)

        model.eval()
        valResult = doValidation(model, dataloaderVal, parallel = parallel, device = device)

        nll = valResult["meanNLL"]
        f1 = valResult["f1"]

        # lrScheduler.step(nll)
        torch.cuda.empty_cache()


        if workerId == 0:
            lossAveraged = sum(lossAll)/len(lossAll)
            lossAll = []
            previousBestF1 = max(lossTracker['val'], default=float("-inf"))
            lossTracker['train'].append(lossAveraged)
            lossTracker['val'].append(f1)

            print(
                    "Validation result (epoch {}, step {}/{}): {}".format(
                            epoc + 1, globalStep, args.nIter, valResult))

            for key in valResult:
                writer.add_scalar('val/'+ key, valResult[key], epoc)

            if f1 >= previousBestF1:
                best_state_dict = copy.deepcopy(model.state_dict())
                print(
                        "Best checkpoint updated at step {} "
                        "(validation F1={:.4f}).".format(globalStep, f1))

            save_checkpoint(
                    filename,
                    checkpoint_resume_epoch(epoc, epoch_complete=True),
                    globalStep, model, lossTracker,
                    best_state_dict, optimizer, lrScheduler,
                    metadata=runMetadata,
                    training_state=build_training_state(
                            checkpoint_resume_epoch(epoc, epoch_complete=True),
                            0, gradNormHist, []))

        if trainingComplete:
            break

    if workerId == 0:
        writer.close()
    if parallel:
        dist.destroy_process_group()


    

if __name__ == '__main__':


    parser = argparse.ArgumentParser("Perform Training")
    parser.add_argument('saved_filename')
    parser.add_argument('--nProcess', help="# of processes for parallel training", required = True, type=int)
    parser.add_argument('--master_addr', help="master address for distributed training ", 
                                         default = '127.0.0.1')

    parser.add_argument('--master_port', help='master port number for distributed training', default = "29500")
    parser.add_argument('--allow_tf32', action = "store_true")
    parser.add_argument('--datasetPath', required = True)
    parser.add_argument('--datasetMetaFile_train', required = True)
    parser.add_argument('--datasetMetaFile_val', required = True)
    parser.add_argument(
            '--noPedalExtension', dest='pedalExtension', action='store_false',
            help="declare that the supplied metadata does not extend note offsets by sustain pedal")
    parser.set_defaults(pedalExtension=True)

    parser.add_argument('--batchSize', default=4, type=int)
    parser.add_argument('--hopSize', required=False, type=float)
    parser.add_argument('--chunkSize', required=False, type=float)
    parser.add_argument(
            '--dataLoaderWorkers', default=0, type=int,
            help=("number of DataLoader worker processes; defaults to 0 and an "
                  "explicit value is honored on every operating system"))
    parser.add_argument('--gradClippingQuantile', default = 0.8, type=float)


    parser.add_argument('--max_lr', default = 2e-4, type=float)
    parser.add_argument('--weight_decay', default = 1e-4, type=float)
    parser.add_argument('--nIter', default= 180000,type = int)
    parser.add_argument(
            '--seed', required=True, type=int,
            help="declared run seed used for model, data-order, and worker RNGs")
    parser.add_argument('--modelConf', required=True, help = "the path to the model conf file")
    parser.add_argument(
            '--systemId', required=True,
            help="protocol system identifier, for example B, C, or D")
    parser.add_argument(
            '--pretrainedCheckpoint',
            help="initialize compatible model weights from a pretrained checkpoint")
    parser.add_argument(
            '--strictPretrained', action="store_true",
            help="require an exact state-dict match when initializing")
    parser.add_argument('--augment',  action ="store_true", help="do data augmentation")
    parser.add_argument('--noiseFolder',  required = False)
    parser.add_argument('--irFolder',  required = False)



    args = parser.parse_args()

    num_processes = args.nProcess
    saved_filename = args.saved_filename

    if args.nIter < 1:
        parser.error("--nIter must be a positive number of optimizer steps")
    if args.dataLoaderWorkers < 0:
        parser.error("--dataLoaderWorkers cannot be negative")
    if args.nProcess != 1:
        parser.error(
                "controlled exact-resume training currently requires --nProcess 1")
    if args.pretrainedCheckpoint is not None:
        if not os.path.isfile(args.pretrainedCheckpoint):
            parser.error(
                    f"Pretrained checkpoint not found: {args.pretrainedCheckpoint}")
        if os.path.exists(saved_filename):
            parser.error(
                    "Pretrained initialization requires a new training "
                    f"checkpoint path: {saved_filename}")
    elif args.strictPretrained:
        parser.error("--strictPretrained requires --pretrainedCheckpoint")

    runSeed = args.seed


    if num_processes == 1:
        train(0, 1, saved_filename, runSeed, args)
    else:
        mp.spawn(fn=train, args=(num_processes, saved_filename, runSeed, args),  nprocs = num_processes, join=True, daemon=False)
