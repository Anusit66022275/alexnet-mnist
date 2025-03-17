import torch.nn as nn
import torchvision.models as models

def get_alexnet():
    alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    alexnet.classifier[6] = nn.Linear(4096, 10)
    return alexnet