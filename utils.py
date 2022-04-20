import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
For our SL, we set A = -4 for all datasets,
and alpha = 0.01, beta = 1.0 for MNIST, alpha = 0.1, beta = 1.0
for CIFAR-10, alpha = 6.0, beta = 0.1 for CIFAR-100 (a dataset
known for hard convergence)
"""
class SCELoss(nn.Module):
    def __init__(self, num_classes=10, alpha=1, beta=1, A=np.exp(-4.0)):
        super(SCELoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.A = A
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=self.A, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.alpha * ce + self.beta * rce.mean()

        return loss