import argparse

from os import listdir
from os.path import join

import PIL.Image as pil_image
import pandas as pd

from model import AutoEncoder, AutoEncoderAuxiliary
from utils import *

import torch
from torch import nn
from torchvision import transforms

from tqdm import tqdm

def main() :
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelWeightsDir", type = str, required = True)
    parser.add_argument("--auxiliaryLoss", action = "store_true")
    parser.add_argument("--testNoisyDir", type = str, required = True)
    parser.add_argument("--testCleanDir", type = str, required = True)
    parser.add_argument("--testSaveDir", type = str, required = True)
    parser.add_argument("--imageChannel", type = int, default = 3)
    args = parser.parse_args()

    # Get Current Namespace
    print(args)

    # Load Trained Model
    if args.auxiliaryLoss :
        model = AutoEncoderAuxiliary(args.imageChannel, args.imageChannel, 64)
        model.load_state_dict(torch.load(args.modelWeightsDir))
    else :
        model = AutoEncoder(args.imageChannel, args.imageChannel, 64)
        model.load_state_dict(torch.load(args.modelWeightsDir))

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

    # Create Torchvision Transforms Instance
    toTensor = transforms.ToTensor()
    toPIL = transforms.ToPILImage()

    # Create List Instance for Saving Results
    imageNameList, PSNRList, SSIMList = list(), list(), list()
    
    # Evaluate Model
    model.eval()

    # Load Image Path
    noisyPathList = [join(args.testNoisyDir, image) for image in listdir(args.testNoisyDir)]
    cleanPathList = [join(args.testCleanDir, image) for image in listdir(args.testCleanDir)]
    
    # Evaluation
    with tqdm(total = len(noisyPathList)) as pBar : 
        with torch.no_grad() :
            for i in range(len(noisyPathList)) :
                # Load Image as Pillow Format
                imageNoisy = pil_image.open(noisyPathList[i])
                imageClean = pil_image.open(cleanPathList[i])
    
                # Convert Pillow Format into PyTorch Tensor + Add Dimension
                tensorNoisy = toTensor(imageNoisy).unsqueeze(0)
                tensorClean = toTensor(imageClean).unsqueeze(0)

                # Assign Device
                if cudaAvailability :
                    tensorNoisy = tensorNoisy.cuda()
      
                # Get Prediction
                if args.auxiliaryLoss :
                    tensorAux, tensorPred = model(tensorNoisy)
                    tensorAux, tensorPred = tensorAux.detach().cpu(), tensorPred.detach().cpu()
                else :
                    tensorPred = model(tensorNoisy).detach().cpu()

                # Calculate PSNR
                psnr = calcPSNR(tensorClean, tensorPred)

                # Calculate SSIM
                ssim = calcSSIM(tensorClean, tensorPred)

                # Append Image Name
                imageNameList.append(f"test_sample{i}.png")

                # Append PSNR
                PSNRList.append(psnr.item())

                # Append SSIM
                SSIMList.append(ssim.item())

                # Convert PyTorch Tensor to Pillow Image Format
                tensorPred = torch.clamp(tensorPred.squeeze(0), min = 0.0, max = 1.0)
                imagePred = toPIL(tensorPred)

                imagePred.save(f"{args.testSaveDir}/test_sample{i}.png")
                
                # Update TQDM Bar
                pBar.update()
                
    # Create Dictionary Instance for Saving Result
    d = {"Denoised Image PSNR(dB)" : PSNRList,
           "Denoised Image SSIM" : SSIMList}

    # Create Pandas DataFrame Instance
    df = pd.DataFrame(data = d, index = imageNameList)

    # Save Result as CSV Format
    df.to_csv(f"{args.testSaveDir}/image_quality_assessment.csv")

if __name__ == "__main__" :
    main()