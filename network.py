import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class Classifier(nn.Module):
    """general clasifier model
    """
    def __init__(self, nClass, inChannels=3, encoder_name='resnet18', pretrained='imagenet', freezeCNN=False):
        """initialization

        Args:
            nClass (int): number of classes 
            inChannels (int, optional): input image channels. Defaults to 3.
            encoder_name (str, optional): encoder/classifier name. Defaults to 'resnet18'.
            pretrained (str, optional): pretrained model weights. Defaults to 'imagenet'.
            freezeCNN (bool, optional): whether freeze cnn layer. Defaults to False.
        """
        super().__init__()
        # load SMP model
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=pretrained,
            in_channels=inChannels,
            classes=nClass,
        )
        self.encoder = model.encoder # extract the model's encoder for feature eactraction

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # average pooling layer
        self.featDims = self.encoder.out_channels[-1] # cnn feature dimension
        self.fc = nn.Linear(self.featDims, nClass) # fully connected layer

        self.freezeCNN = freezeCNN # whether freeze the CNN for transfer learning

    def forward(self, X):
        feat = self.extractFeature(X) # extract feature
        # classification
        if self.freezeCNN:
            y = self.fc(feat.detach())
        else:
            y = self.fc(feat)

        return y

    def extractFeature(self, X):
        feat = self.encoder(X) # encode
        feat = feat[-1] # extract last feature
        feat = self.avgpool(feat) # average pooling
        feat = feat.view(X.shape[0], -1) # adapt dimensions

        return feat
