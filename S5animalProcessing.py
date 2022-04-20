import json
import os

import numpy as np
from cleanlab.pruning import get_noise_indices
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from network import *
from trainers import *


#%%
class Animal(Dataset):
    def __init__(self, filenames, labels, mode='train'):
        self.filenames = filenames
        self.labels = labels
        self.targetsize = [64, 64]
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img = Image.open(self.filenames[index])
        img = img.resize(self.targetsize)
        img = self.transform(img)

        target = self.labels[index]

        return img, target, self.filenames[index]


#%% Configs
nClass = 10
epochs = 10  # train epochs
batchSize = 512  # batch size
learning_rate = 1e-3  # learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%% process all the file
trainfiles, trainlabels = [], []
trainpath = os.path.join('Datasets', 'animal_10n', 'training')
for root, folder, files in os.walk(trainpath):
    for f in files:
        label = int(f[0])
        trainlabels.append(label)
        trainfiles.append(os.path.join(root, f))
trainfiles = np.array(trainfiles)
trainlabels = np.array(trainlabels)
index = np.arange(len(trainlabels))
np.random.shuffle(index)
trainfiles = trainfiles[index]
trainlabels = trainlabels[index]

testfiles, testlabels = [], []
testpath = os.path.join('Datasets', 'animal_10n', 'testing')
for root, folder, files in os.walk(testpath):
    for f in files:
        label = int(f[0])
        testlabels.append(label)
        testfiles.append(os.path.join(root, f))

testfiles = np.array(testfiles)
testlabels = np.array(testlabels)
#%% generate dataloaders for cross validation
trainDataLoaders, validDataLoaders = [], []
kf = KFold(5)  # K fold
for trainIdx, validIdx in kf.split(trainfiles):
    # train datast depend on train Index
    dataset = Animal(trainfiles[trainIdx], trainlabels[trainIdx], mode='train')
    trainDataLoaders.append(DataLoader(dataset, batch_size=batchSize, shuffle=True))

    # validation dataset depend on validation index
    dataset = Animal(trainfiles[validIdx], trainlabels[validIdx], mode='valid')
    validDataLoaders.append(DataLoader(dataset, batch_size=batchSize, shuffle=False))

#%% cross validtion train
filesall, yall, probas = [], [], []
for k in range(len(trainDataLoaders)):
    print('Cross validation kfold: ', k + 1)
    # data
    trainDataLoader = trainDataLoaders[k]
    validDataLoader = validDataLoaders[k]

    # model
    model = Classifier(nClass, inChannels=3, encoder_name='resnet18', pretrained='imagenet')
    # initialize trainer
    trainer = Trainer(model, learning_rate=learning_rate, epochs=epochs, evalEnable=False)
    # train the model
    trainer.fit(trainDataLoader=trainDataLoader, testDataLoader=None)
    # predict probability on validation set
    model = trainer.model
    model.eval()

    # collect validation data's probability
    for batch_data in validDataLoader:
        xData = batch_data[0].float().to(device)  # load to device
        with torch.no_grad():
            pred = model(xData)
            pred = torch.softmax(pred, dim=-1)
        probas.append(pred.detach().cpu().numpy())
        filesall.append(batch_data[-1])
        yall.append(batch_data[1])

filesall = np.concatenate(filesall, axis=0)
yall = np.hstack(yall)
probas = np.vstack(probas)
# save all the result for next step
np.savez('./Datasets/animal_noisy.npz', X=filesall, y=yall, probas=probas)

#===============================================================================

