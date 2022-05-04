import torch
from torch import nn

class Conv2dLayer(nn.Module) :
    def __init__(self, inChannels, outChannels, kernelSize, stride, dilation = 1, bias = False, norm = "in", act = "leaky") :
        # Inheritance
        super(Conv2dLayer, self).__init__()
        
        # Create Layer Instance
        self.conv = nn.Conv2d(
                      in_channels = inChannels,
                      out_channels = outChannels,
                      kernel_size = kernelSize,
                      stride = stride,
                      padding = (dilation * (kernelSize - 1)) // 2 ,
                      dilation = dilation,
                      bias = bias
                      )

        # Create Normalization Instance
        if norm == "bn" :
            self.norm = nn.BatchNorm2d(outChannels)
        elif norm == "in" :
            self.norm = nn.InstanceNorm2d(outChannels)
        
        # Create Activation Function Instance
        if act == "relu" :
            self.act = nn.ReLU(inplace = True)
        elif act == "leaky" :
            self.act = nn.LeakyReLU(0.2, inplace = True)
        elif act == "elu" :
            self.act = nn.ELU(inplace = True)
        elif act == "selu" :
            self.act = nn.SELU(inplace = True)
        elif act == "silu" :
            self.act = nn.SiLU(inplace = True)
        elif act == "prelu" :
            self.act = nn.PReLU()
        elif act == "hardswish" :
            self.act = nn.Hardswish(inplace = True)
        elif act == "gelu" :
            self.act = nn.GELU()
        elif act == "sigmoid" :
            self.act = nn.Sigmoid()
    
    def forward(self, input) :
        output = self.conv(input)
        
        if hasattr(self, "norm") :
            output = self.norm(output)
        
        if hasattr(self, "act") :
            output = self.act(output)
            
        return output

class EncoderResidualBlock(nn.Module) :
    def __init__(self, channels, kernelSize, stride, dilation = 1, bias = False, norm = "in", act = "leaky") :
        # Inheritance
        super(EncoderResidualBlock, self).__init__()
        
        # Create Layer Instance
        self.convIn = Conv2dLayer(channels, channels, kernelSize, stride, dilation, bias, norm, act)
        self.convOut = Conv2dLayer(channels, channels, kernelSize, stride, dilation, bias, norm, act)
        
    def forward(self, input) :
        output = self.convIn(input)
        output = self.convOut(output)
        output = output + input
        
        return output

class DecoderResidualBlock(nn.Module) :
    def __init__(self, channels, kernelSize, stride, dilation = 1, bias = False, norm = "in", act = "leaky") :
        # Inheritance
        super(DecoderResidualBlock, self).__init__()
        
        # Create Layer Instance
        self.convIn = Conv2dLayer(channels * 2, channels, kernelSize, stride, dilation, bias, norm, act)
        self.convOut = Conv2dLayer(channels, channels, kernelSize, stride, dilation, bias, norm, act)
        self.bottleneck = Conv2dLayer(channels * 2, channels, 1, 1, 1, bias, None, None)
        
    def forward(self, input) :
        output = self.convIn(input)
        output = self.convOut(output)
        output = output + self.bottleneck(input)
        
        return output

class Downsample(nn.Module) :
    def __init__(self, numLayer, inChannels, outChannels, kernelSize, stride, dilation = 1, bias = False, norm = "in", act = "leaky") :
        # Inheritance
        super(Downsample, self).__init__()
        
        # Create Layer Instance
        self.down = nn.Sequential(
                          Conv2dLayer(inChannels, outChannels, kernelSize, 2, dilation, bias, norm, None),
                          *[EncoderResidualBlock(outChannels, kernelSize, stride, dilation, bias, norm, act) for _ in range(numLayer)]
                          )

    def forward(self, input) :
        output = self.down(input)
        
        return output
    
class Upsample(nn.Module) :
    def __init__(self, inChannels, outChannels, kernelSize, stride, dilation = 1, bias = False, norm = "in", act = "leaky") :
        # Inheritance
        super(Upsample, self).__init__()
        
        # Create Layer Instance
        self.up = nn.Sequential(
                    nn.Upsample(scale_factor = 2, mode = "bicubic"),
                    Conv2dLayer(inChannels, outChannels, kernelSize, stride, dilation, bias, norm, act)
                    )
        self.ResBlock = DecoderResidualBlock(outChannels, kernelSize, stride, dilation, bias, norm, act)
        
    def forward(self, input, skipConnection) :
        output = self.up(input)
        output = torch.cat([output, skipConnection], dim = 1)
        output = self.ResBlock(output)
        
        return output
