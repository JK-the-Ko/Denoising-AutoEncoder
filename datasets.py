from os import listdir
from os.path import join

from random import randint

import PIL.Image as pil_image

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DataFromFolder(Dataset) :
    def __init__(self, noisyImageDir, cleanImageDir, mode) :
        # Inheritance
        super(DataFromFolder, self).__init__()

        # Load Image Path
        self.noisyImagePathList = [join(noisyImageDir, image) for image in listdir(noisyImageDir)]
        self.cleanImagePathList = [join(cleanImageDir, image) for image in listdir(cleanImageDir)]
        
        # Initialize Probability for Data Augmentation
        probHorizontal = randint(0, 1)
        probVertical = randint(0, 1)

        # Create Torchvision Transforms Instance
        if mode == "train" :
            # Training Phase Torchvision Transforms Instance
            self.transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.RandomHorizontalFlip(p = probHorizontal),
                                 transforms.RandomVerticalFlip(p = probVertical),
                                 ])

        elif mode == "valid" :
            # Validation Phase Torchvision Transforms Instance
            self.transform = transforms.ToTensor()

    def loadImage(self, imagePath) :
        # Load Image as Pillow Format
        image = pil_image.open(imagePath)
        
        return image

    def __getitem__(self, index) :
        # Load Image
        input = self.loadImage(self.noisyImagePathList[index])
        target = self.loadImage(self.cleanImagePathList[index])

        # Apply Torchvision Transforms
        input = self.transform(input)
        target = self.transform(target)

        return input, target

    def __len__(self) :
        # Get Number of Images
        return len(self.noisyImagePathList)