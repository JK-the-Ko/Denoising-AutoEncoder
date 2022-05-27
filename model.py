from torch import nn

from modelParts import Conv2dLayer, EncoderResidualBlock, Downsample, Upsample

class AutoEncoder(nn.Module) :
    def __init__(self, inChannels, outChannels, channels) :
        # Inheritance
        super(AutoEncoder, self).__init__()
        
        # Create Layer Instance
        self.convIn = nn.Sequential(
                            Conv2dLayer(inChannels, channels, 7, 1, norm = None, act = None),
                            *[EncoderResidualBlock(channels, 3, 1, norm = None) for _ in range(4)]
                            )
        self.convOut = Conv2dLayer(channels, outChannels, 7, 1, norm = None, act = None)
        self.down1 = Downsample(4, channels, channels, 3, 1, norm = None)
        self.down2 = Downsample(4, channels, channels, 3, 1, norm = None)
        self.down3 = Downsample(4, channels, channels, 3, 1, norm = None)
        self.ResBlock = nn.Sequential(*[EncoderResidualBlock(channels, 3, 1, norm = None) for _ in range(5)])
        self.up3 = Upsample(channels, channels, 3, 1, norm = None)
        self.up2 = Upsample(channels, channels, 3, 1, norm = None)
        self.up1 = Upsample(channels, channels, 3, 1, norm = None)
        
        # Initialize Parameter Weights
        self.initializeWeights()
        
    def forward(self, input) :
        skipConnection = list()
        
        output = self.convIn(input)
        skipConnection.append(output)
        output = self.down1(output)
        skipConnection.append(output)
        output = self.down2(output)
        skipConnection.append(output)
        output = self.down3(output)
        output = self.ResBlock(output)
        output = self.up3(output, skipConnection.pop())
        output = self.up2(output, skipConnection.pop())
        output = self.up1(output, skipConnection.pop())
        output = self.convOut(output)
        output = output + input
        
        return output
    
    def initializeWeights(self) :
        for m in self.modules() :
            if isinstance(m, nn.Conv2d) :
                # Apply Xavier Uniform Initialization
                nn.init.xavier_uniform_(m.weight.data)

                if m.bias is not None :
                    # Set Bias to Zero
                    m.bias.data.zero_()

class AutoEncoderAuxiliary(nn.Module) :
    def __init__(self, inChannels, outChannels, channels) :
        # Inheritance
        super(AutoEncoderAuxiliary, self).__init__()
        
        # Create Layer Instance
        self.convIn = nn.Sequential(
                            Conv2dLayer(inChannels, channels, 7, 1, norm = None, act = None),
                            *[EncoderResidualBlock(channels, 3, 1, norm = None) for _ in range(4)]
                            )
        self.convOut = Conv2dLayer(channels, outChannels, 7, 1, norm = None, act = None)
        self.auxOut = nn.Sequential(
                          nn.Upsample(scale_factor = 8, mode = "bicubic"),
                          Conv2dLayer(channels, outChannels, 3, 1, norm = None, act = None)
                          )
        self.down1 = Downsample(4, channels, channels, 3, 1, norm = None)
        self.down2 = Downsample(4, channels, channels, 3, 1, norm = None)
        self.down3 = Downsample(4, channels, channels, 3, 1, norm = None)
        self.ResBlock = nn.Sequential(*[EncoderResidualBlock(channels, 3, 1, norm = None) for _ in range(5)])
        self.up3 = Upsample(channels, channels, 3, 1, norm = None)
        self.up2 = Upsample(channels, channels, 3, 1, norm = None)
        self.up1 = Upsample(channels, channels, 3, 1, norm = None)
        
        # Initialize Parameter Weights
        self.initializeWeights()
        
    def forward(self, input) :
        skipConnection = list()
        
        output = self.convIn(input)
        skipConnection.append(output)
        output = self.down1(output)
        skipConnection.append(output)
        output = self.down2(output)
        skipConnection.append(output)
        output = self.down3(output)
        output = self.ResBlock(output)
        outputAux = self.auxOut(output)
        output = self.up3(output, skipConnection.pop())
        output = self.up2(output, skipConnection.pop())
        output = self.up1(output, skipConnection.pop())
        output = self.convOut(output)
        output = output + input
        
        return outputAux, output
    
    def initializeWeights(self) :
        for m in self.modules() :
            if isinstance(m, nn.Conv2d) :
                # Apply Xavier Uniform Initialization
                nn.init.xavier_uniform_(m.weight.data)

                if m.bias is not None :
                    # Set Bias to Zero
                    m.bias.data.zero_()