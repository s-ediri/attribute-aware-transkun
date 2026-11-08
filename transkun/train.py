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


def seed_everything(seed, worker_id=0):
    """Seed one training process deterministically from a declared run seed."""
    process_seed = int(seed) + int(worker_id)
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    return process_seed


def train(workerId, nWorker, filename, runSeed, args):
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
                                pretrainedCheckpoint=args.pretrainedCheckpoint)


            save_checkpoint(filename, startEpoch, startIter, model,  lossTracker, best_state_dict, optimizer, lrScheduler)

    if parallel:
        dist.barrier()
        

    startEpoch, startIter, model,  lossTracker, best_state_dict, optimizer, lrScheduler= load_checkpoint(TransKun, conf, filename,device)
    print(
            "Worker {} loaded checkpoint at step {}/{}.".format(
                    workerId, startIter, args.nIter))
    if startIter >= args.nIter:
        if workerId == 0:
            print(f"Training already reached the requested {args.nIter} steps.")
        if parallel:
            dist.destroy_process_group()
        return

    if workerId == 0:
        print("Loading training and validation datasets...")

    datasetPath = args.datasetPath
    datasetPicklePath = args.datasetMetaFile_train
    datasetPicklePath_val = args.datasetMetaFile_val

    dataset = Data.DatasetMaestro(datasetPath, datasetPicklePath)
    datasetVal = Data.DatasetMaestro(datasetPath, datasetPicklePath_val)

    print("Worker {} loaded datasets.".format(workerId))


    if workerId == 0:
        writer = SummaryWriter(filename+".log")

    globalStep = startIter
    # create dataloader


    # this iterator should be constructed each time
    batchSize = args.batchSize

    if args.hopSize is None:
        hopSize = conf.segmentHopSizeInSecond
    else:
        hopSize = args.hopSize

    if args.chunkSize is None:
        chunkSize =  conf.segmentSizeInSecond
    else:
        chunkSize = args.chunkSize

    gradNormHist = MovingBuffer(initValue = 40, maxLen = 10000)

    augmentator = None
    if args.augment:
        augmentator = Data.AugmentatorAudiomentations(sampleRate=44100, noiseFolder = args.noiseFolder, convIRFolder = args.irFolder)

    dataLoaderOptions = {"num_workers": args.dataLoaderWorkers}
    if args.dataLoaderWorkers > 0:
        dataLoaderOptions["prefetch_factor"] = max(4, args.dataLoaderWorkers)

    trainingComplete = False
    for epoc in range(startEpoch, 1000000):
        # average length will be chunkSize
        dataIter = Data.DatasetMaestroIterator(dataset, hopSize, chunkSize, seed = epoc*100+runSeed, augmentator = augmentator, notesStrictlyContained=False)

        if parallel:
            sampler = torch.utils.data.distributed.DistributedSampler(dataIter)
            sampler.set_epoch(epoc)
            # dataloader= torch.utils.data.DataLoader(dataIter, batch_size = batchSize, collate_fn = Data.collate_fn, num_workers=args.dataLoaderWorkers, sampler = sampler, drop_last = True, prefetch_factor = 3)
            dataloader= torch.utils.data.DataLoader(dataIter, batch_size = batchSize, collate_fn = Data.collate_fn_batching, sampler = sampler, drop_last = True, **dataLoaderOptions)
        else:
            dataloader= torch.utils.data.DataLoader(dataIter, batch_size = batchSize, collate_fn = Data.collate_fn_batching, shuffle=True, drop_last=True, **dataLoaderOptions)


        lossAll = []

        # curLRScheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 5e-5, max_lr = 1e-4, step_size_up = len(dataloader)//10*5, step_size_down = len(dataloader)//10*5, cycle_momentum=False)
        globalStepWarmupCutoff = globalStep+500

        for idx, batch in enumerate(dataloader):
            if workerId ==0:
                currentLR = [p["lr"] for p in optimizer.param_groups][0]
                writer.add_scalar(f'Optimizer/lr', currentLR , globalStep)

            computeStats = False 
            if idx % 40 == 0:
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
                if globalStep> globalStepWarmupCutoff:
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
                                100 * (idx + 1) / len(dataloader),
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


                if idx%2000 ==1999:
                    save_checkpoint(filename, epoc+1, globalStep+1, model,  lossTracker, best_state_dict, optimizer, lrScheduler)
                    print(
                            "Checkpoint saved at step {}/{}: {}".format(
                                    globalStep + 1, args.nIter, filename))

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
            dataloaderVal= torch.utils.data.DataLoader(dataIterVal, batch_size = 2*batchSize, collate_fn = Data.collate_fn , num_workers=args.dataLoaderWorkers, sampler = samplerVal)
        else:
            dataloaderVal= torch.utils.data.DataLoader(dataIterVal, batch_size = 2*batchSize, collate_fn = Data.collate_fn , num_workers=args.dataLoaderWorkers, shuffle=True)

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

            save_checkpoint(filename, epoc+1, globalStep, model,  lossTracker, best_state_dict, optimizer, lrScheduler)

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

    parser.add_argument('--batchSize', default=4, type=int)
    parser.add_argument('--hopSize', required=False, type=float)
    parser.add_argument('--chunkSize', required=False, type=float)
    parser.add_argument('--dataLoaderWorkers', default = 2, type=int)
    parser.add_argument('--gradClippingQuantile', default = 0.8, type=float)


    parser.add_argument('--max_lr', default = 2e-4, type=float)
    parser.add_argument('--weight_decay', default = 1e-4, type=float)
    parser.add_argument('--nIter', default= 180000,type = int)
    parser.add_argument(
            '--seed', required=True, type=int,
            help="declared run seed used for model, data-order, and worker RNGs")
    parser.add_argument('--modelConf', required=True, help = "the path to the model conf file")
    parser.add_argument(
            '--pretrainedCheckpoint',
            help="initialize compatible model weights from a pretrained checkpoint")
    parser.add_argument('--augment',  action ="store_true", help="do data augmentation")
    parser.add_argument('--noiseFolder',  required = False)
    parser.add_argument('--irFolder',  required = False)



    args = parser.parse_args()

    num_processes = args.nProcess
    saved_filename = args.saved_filename

    if args.nIter < 1:
        parser.error("--nIter must be a positive number of optimizer steps")
    if args.pretrainedCheckpoint is not None:
        if not os.path.isfile(args.pretrainedCheckpoint):
            parser.error(
                    f"Pretrained checkpoint not found: {args.pretrainedCheckpoint}")
        if os.path.exists(saved_filename):
            parser.error(
                    "Pretrained initialization requires a new training "
                    f"checkpoint path: {saved_filename}")

    runSeed = args.seed


    if num_processes == 1:
        train(0, 1, saved_filename, runSeed, args)
    else:
        mp.spawn(fn=train, args=(num_processes, saved_filename, runSeed, args),  nprocs = num_processes, join=True, daemon=False)
