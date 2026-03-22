import torch.nn as nn
import torchvision.models as models

class SkinCNN(nn.Module):

    def __init__(self,num_classes):
        super(SkinCNN,self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')

        self.model.fc = nn.Linear(
            self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )
        )

    def forward(self,x):
        return self.model(x)