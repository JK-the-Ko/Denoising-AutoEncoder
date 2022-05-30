import argparse
import copy

from os import listdir, getcwd, mkdir
from os.path import join

import wandb

from datasets import DataFromFolder
from model import AutoEncoder, AutoEncoderAuxiliary
from loss import *
from utils import *

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from torchinfo import summary

from tqdm import tqdm

def main() :
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type = str, required = True)
    parser.add_argument("--modelName", type = str, default = "DAE")
    parser.add_argument("--auxiliaryLoss", action = "store_true")
    parser.add_argument("--SSIMLoss", action = "store_true")
    parser.add_argument("--trainNoisyDir", type = str, required = True)
    parser.add_argument("--trainCleanDir", type = str, required = True)
    parser.add_argument("--validNoisyDir", type = str, required = True)
    parser.add_argument("--validCleanDir", type = str, required = True)
    parser.add_argument("--trainSize", type = int, default = 64)
    parser.add_argument("--validSize", type = int, default = 256)
    parser.add_argument("--imageChannel", type = int, default = 3)
    parser.add_argument("--batchSize", type = int, default = 8)
    parser.add_argument("--epochs", type = int, default = 100)
    parser.add_argument("--lr", type = float, default = 4e-4)
    args = parser.parse_args()

    # Get Current Namespace
    print(args)

    # Initialize Weights & Biases Library
    wandb.init(config = args, resume = "never", project = args.project)

    # Initialize Project Name
    if args.auxiliaryLoss :
        if args.SSIMLoss :
            args.modelName += "-Aux-L1+SSIM"
        else :
            args.modelName += "-Aux-L1"

    else :
        if args.SSIMLoss :
            args.modelName += "-Vanilla-L1+SSIM"
        else :
            args.modelName += "-Vanilla-L1"
    
    wandb.run.name = args.modelName
        
    # Set Seed
    setSeed(1)
    # Create model Instance
    if args.auxiliaryLoss :
        model = AutoEncoderAuxiliary(args.imageChannel, args.imageChannel, 64)
    else :
        model = AutoEncoder(args.imageChannel, args.imageChannel, 64)

    # Set Seed
    setSeed(1)
    # Create DataFromFolder Instance
    trainDataset = DataFromFolder(args.trainNoisyDir, args.trainCleanDir, "train")
    validDataset = DataFromFolder(args.validNoisyDir, args.validCleanDir, "valid")

    # Create DataLoader Instance
    trainDataloader = DataLoader(trainDataset, batch_size = args.batchSize, shuffle = True, drop_last = True)
    validDataloader = DataLoader(validDataset, batch_size = args.batchSize, shuffle = False, drop_last = True)

    # Create Optimizer Instance
    optimizer = optim.Adam(params = model.parameters(), lr = args.lr, weight_decay = 1e-5)
    
    # Watch Training Process
    wandb.watch(model)

    # Create Learning Rate Scheduler Instance
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = args.epochs, eta_min = args.lr * 1e-2)
    
    # Check CUDA Availability
    cudaAvailability = torch.cuda.is_available()
    
    # Assign Device
    if cudaAvailability :
        # Multi-GPU Environment
        if torch.cuda.device_count() > 1 :
            model = nn.DataParallel(model).cuda()
        else :
            # Single-GPU Environment
            model = model.cuda()
    
    # Summarize Model
    summary(model, (args.batchSize, args.imageChannel, args.trainSize, args.trainSize))
    
    # Create Directory for Saving Weights
    if "best_model" not in listdir(getcwd()) :
        mkdir(join(getcwd(), "best_model"))
    if "latest_model" not in listdir(getcwd()) :
        mkdir(join(getcwd(), "latest_model"))
    
    if args.project not in listdir(join(getcwd(), "best_model")) :
        mkdir(join(getcwd(), "best_model", args.project))
    if args.project not in listdir(join(getcwd(), "latest_model")) :
        mkdir(join(getcwd(), "latest_model", args.project))
  
    # Initialize Metric
    bestEpoch = 0
    bestPSNR, bestSSIM = 0, 0
    
    # Training
    for epoch in range(args.epochs) :
        # Get Current Learning Rate
        for paramGroup in optimizer.param_groups:
            currentLR = paramGroup["lr"]

        # Create TQDM Instance
        trainBar = tqdm(trainDataloader)

        # Train Model
        model.train()

        # Create AverageMeter Instance for Saving Result
        trainLoss = AverageMeter()
        trainPSNR, trainSSIM = AverageMeter(), AverageMeter()

        # Mini-Batch Training
        for data in trainBar :
            # Load Data
            inputs, targets = data

            # Assign Device
            if cudaAvailability :
                inputs = inputs.cuda()
                targets = targets.cuda()
            
            # Set Gradient to Zero
            optimizer.zero_grad()
            
            # Check Model
            if args.auxiliaryLoss :
                # Forward Pass Input Data
                predsAux, preds = model(inputs)
                
                # Calculate Loss
                if args.SSIMLoss :
                    loss = F.l1_loss(predsAux, targets) + F.l1_loss(preds, targets) + SSIMLoss(preds, targets)
                else :
                    loss = F.l1_loss(predsAux, targets) + F.l1_loss(preds, targets)
            else :
                # Forward Pass Input Data
                preds = model(inputs)

                # Calculate Loss
                if args.SSIMLoss :
                    loss = F.l1_loss(preds, targets) + SSIMLoss(preds, targets)
                else :
                    loss = F.l1_loss(preds, targets)
        
            # Calculate Gradient
            loss.backward()
            
            # Update Generator
            optimizer.step()
            
            # Update Loss
            trainLoss.update(loss.item(), len(inputs))
            
            # Update PSNR
            trainPSNR.update(calcPSNR(preds, targets).item(), len(inputs))
            
            # Update SSIM
            trainSSIM.update(calcSSIM(preds, targets).item(), len(inputs))

            # Update TQDM Bar
            trainBar.set_description(desc=f"[{epoch}/{args.epochs - 1}] [Train] [Loss : {trainLoss.avg:.6f}, PSNR(SR) : {trainPSNR.avg:.6f}, SSIM(SR) : {trainSSIM.avg:.6f}]")

        # Create TQDM Instance
        validBar = tqdm(validDataloader)

        # Validate Model
        model.eval()
        
        # Create AverageMeter Instance for Saving Result
        validLoss = AverageMeter()
        validPSNR, validSSIM = AverageMeter(), AverageMeter()

        with torch.no_grad() :
            # Mini-Batch Validation
            for data in validBar :
                # Load Data
                inputs, targets = data

                # Assign Device
                if cudaAvailability :
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                # Check Model
                if args.auxiliaryLoss :
                    # Forward Pass Input Data
                    predsAux, preds = model(inputs)
                    
                    # Calculate Loss
                    if args.SSIMLoss :
                        loss = F.l1_loss(predsAux, targets) + F.l1_loss(preds, targets) + SSIMLoss(preds, targets)
                    else :
                        loss = F.l1_loss(predsAux, targets) + F.l1_loss(preds, targets)
                else :
                    # Forward Pass Input Data
                    preds = model(inputs)

                    # Calculate Loss
                    if args.SSIMLoss :
                        loss = F.l1_loss(preds, targets) + SSIMLoss(preds, targets)
                    else :
                        loss = F.l1_loss(preds, targets)

                # Update Loss
                validLoss.update(loss.item(), len(inputs))
                
                # Update PSNR
                validPSNR.update(calcPSNR(preds, targets).item(), len(inputs))
                
                # Update SSIM
                validSSIM.update(calcSSIM(preds, targets).item(), len(inputs))

                # Update TQDM Bar
                validBar.set_description(desc=f"[{epoch}/{args.epochs - 1}] [Validation] [Loss : {validLoss.avg:.6f}, PSNR(SR) : {validPSNR.avg:.6f}, SSIM(SR) : {validSSIM.avg:.6f}]")

        # Create List Instance for Visualizing Result
        sampleList = list()

        # Append Image
        for i in range(args.batchSize) :
            sampleImage = concatenateImage(
                                  torch.clamp(inputs[i].cpu().squeeze(0), min =0.0, max = 1.0),
                                  torch.clamp(preds[i].cpu().squeeze(0), min =0.0, max = 1.0),
                                  torch.clamp(targets[i].cpu().squeeze(0), min = 0.0, max = 1.0)
                                  )

            sampleList.append(wandb.Image(sampleImage, caption = f"Sample {i + 1}"))

        # Update Log
        wandb.log({
            "Learning Rate" : currentLR,
            "Validation PSNR" : validPSNR.avg,
            "Validation SSIM" : validSSIM.avg,
            "Validation Loss" : validLoss.avg,
            "Image Comparison" : sampleList})

        # Check Metric
        if validPSNR.avg > bestPSNR :
            # Save Best Model
            if torch.cuda.device_count() > 1 :
                bestModel = copy.deepcopy(model.module.state_dict())
            else :
                bestModel = copy.deepcopy(model.state_dict())

            # Update Metric
            bestPSNR = validPSNR.avg
            bestEpoch = epoch

            # Save Best Model
            torch.save(bestModel, f"best_model/{args.project}/{args.modelName}_best.pth")

        # Check Metric
        if validSSIM.avg > bestSSIM :
            # Save Best Model
            if torch.cuda.device_count() > 1 :
                bestModel = copy.deepcopy(model.module.state_dict())
            else :
                bestModel = copy.deepcopy(model.state_dict())

            # Update Metric
            bestSSIM = validSSIM.avg
            bestEpoch = epoch

            # Save Best Model
            torch.save(bestModel, f"best_model/{args.project}/{args.modelName}_best.pth")
            
        # Save Lastest Model
        if torch.cuda.device_count() > 1 :
            lastestModel = copy.deepcopy(model.module.state_dict())
        else :
            lastestModel = copy.deepcopy(model.state_dict())
        
        torch.save(lastestModel, f"latest_model/{args.project}/{args.modelName}_latest.pth")

        # Update Learning Rate Scheduler
        scheduler.step()

    # Print Training Result
    print(f"Best Epoch : {bestEpoch}")
    print(f"Best PSNR : {bestPSNR:.6f}")
    print(f"Best SSIM : {bestSSIM:.6f}")

if __name__ == "__main__" :
    main()