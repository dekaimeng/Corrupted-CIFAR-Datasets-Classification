import json
import os

import numpy as np
from cleanlab.pruning import get_noise_indices
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from loader_for_CIFAR import *
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
epochs = 15  # train epochs
batchSize = 512  # batch size
learning_rate = 1e-3  # learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
data = np.load('./Datasets/animal_noisy.npz')
filesall = data['X']
y = data['y']
probas = data['probas']
# clearnlab to detect the noisy labels. Bool true is noisy samples
noisyIndex = get_noise_indices(y, probas, prune_method='both', frac_noise=1, n_jobs=1)
trainclearfiles = filesall[~noisyIndex]  # indice data that is not noisy
trainclearlabels = y[~noisyIndex]  # indice label that is not noisy
print('Num of raw train data: ', len(filesall))
print('Num of cleared train data: ', len(trainclearfiles))

testfiles, testlabels = [], []
testpath = os.path.join('Datasets', 'animal_10n', 'testing')
for root, folder, files in os.walk(testpath):
    for f in files:
        label = int(f[0])
        testlabels.append(label)
        testfiles.append(os.path.join(root, f))

testfiles = np.array(testfiles)
testlabels = np.array(testlabels)

# train datast depend on train Index
dataset = Animal(trainclearfiles, trainclearlabels, mode='train')
trainDataLoader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
# validation dataset depend on validation index
dataset = Animal(testfiles, testlabels, mode='test')
testDataLoader = DataLoader(dataset, batch_size=batchSize, shuffle=False)

#%% train on the cleaned data
# initialize model
model = Classifier(nClass, inChannels=3, encoder_name='densenet121', pretrained='imagenet')
# initialize trainer
trainer = Trainer(model, learning_rate=learning_rate, epochs=epochs, evalEpoch=1)
# train the model
trainer.fit(
    trainDataLoader=trainDataLoader,
    testDataLoader=testDataLoader,
)
# save train logs and weights
np.save('./Results/history_CE_animal.npy', trainer.logger)
torch.save(trainer.model, './Results/model_animal.pt')
