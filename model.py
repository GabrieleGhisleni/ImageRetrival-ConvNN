import torch
import torch.nn as nn
from torchvision import models

def load_model(path, resnet=18):
  class ResNet(nn.Module):
      def __init__(self, num_classes=50, pretrained=True):
          super(ResNet, self).__init__()
          if resnet == 18:
            self.model = models.resnet18(pretrained=pretrained)
          elif resnet==50:
            self.model = models.resnet50(pretrained=pretrained)
          elif resnet==101:
            self.model = models.resnet101(pretrained=pretrained)
          elif resnet==152:
            self.model = models.resnet152(pretrained=pretrained)
          elif resnet == "resnext101_32x8d":
            self.model = models.resnext101_32x8d(pretrained=pretrained)
          elif resnet == "wide_resnet":
            self.model = models.wide_resnet101_2(pretrained=pretrained)
          elif resnet == "google":
            self.model = models.googlenet(pretrained=pretrained)
          num_ftrs = self.model.fc.in_features
          self.model.fc = nn.Sequential(nn.Dropout(0.00005), #to see different results try to change the Dropout domain -> [0,1]
                                        nn.Linear(num_ftrs, num_classes))
      def forward(self, x):
          return self.model(x)

  model = ResNet()
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model_state_dict'])
  return model
