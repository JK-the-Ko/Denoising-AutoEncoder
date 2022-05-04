import argparse

from os import listdir, getcwd, mkdir
from os.path import join

from model import AutoEncoder, AutoEncoderAuxiliary

import torch

from torchinfo import summary

def main() :
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelName", type = str, default = "DAE")
    parser.add_argument("--imageChannel", type = int, default = 3)
    parser.add_argument("--inputSize", type = int, default = 256)
    parser.add_argument("--batchSize", type = int, default = 8)
    args = parser.parse_args()

    # Get Current Namespace
    print(args)

    # Create Model Instance
    AE = AutoEncoder(args.imageChannel, args.imageChannel, channels = 64)
    AEAux = AutoEncoderAuxiliary(args.imageChannel, args.imageChannel, channels = 64)
    
    # Check CUDA Availability
    cudaAvailability = torch.cuda.is_available()
    
    # Assign Device
    if cudaAvailability :
        # Single-GPU Environment
        AE = AE.cuda()
        AEAux = AEAux.cuda()
        
    # Summarize Model
    summary(AE, (args.batchSize, args.imageChannel, args.inputSize, args.inputSize))
    summary(AEAux, (args.batchSize, args.imageChannel, args.inputSize, args.inputSize))

    # Create Dummy Data Instance for Exporting Model
    dummyData = torch.empty(1, args.imageChannel, args.inputSize, args.inputSize, dtype = torch.float32).cuda()

    # Create Directory for Saving ONNX Model
    if "onnx_model" not in listdir(getcwd()) :
        mkdir(join(getcwd(), "onnx_model"))

    # Export Model as ONNX
    torch.onnx.export(AE, dummyData, f"onnx_model/AE.onnx", opset_version = 11)
    torch.onnx.export(AEAux, dummyData, f"onnx_model/AEAux.onnx", opset_version = 11)

if __name__ == "__main__" :
    main()