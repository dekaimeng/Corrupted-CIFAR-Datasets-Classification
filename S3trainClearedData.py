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
dataset = 'cifar100' # dataset name
labelfile = 'cifar100_noisy_labels_task1' # label name
nClass = 10  # number of classes
dataDir = os.path.join('Datasets', 'cifar-100-python')  # data path
noisyLabelFile = os.path.join('.', 'Datasets', '{}.json'.format(labelfile))  # noisy label file

epochs = 15 # train epochs

batchSize = 128 # batch Size
learning_rate = 1e-3 # learning rate

#%% determinde noisy datas
data = np.load(noisyLabelFile.replace('.json', '_noisy.npz'))
X = data['X']
y = data['y']
probas = data['probas']
# clearnlab to detect the noisy labels. Bool true is noisy samples
noisyIndex = get_noise_indices(y, probas, prune_method='both', frac_noise=1, n_jobs=1)
Xtrain = X[~noisyIndex, :] # indice data that is not noisy
ytrain = y[~noisyIndex] # indice label that is not noisy
print('Num of raw train data: ', X.shape[0])
print('Num of cleared train data: ', Xtrain.shape[0])
print("Noisy rate: ", (X.shape[0]-Xtrain.shape[0])/(X.shape[0]))

#%% load cleaned data
loader = cifar_dataloader(dataset, batchSize, 0, dataDir, noisyLabelFile)
transforms_train = loader.transform_train
testDataLoader = loader.run('test')
trainDataset = simple_cifar_dataset(Xtrain, ytrain, transform=transforms_train)
trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)

#%% train on the cleaned data
# initialize model
model = Classifier(nClass, inChannels=3, encoder_name='resnet18', pretrained='imagenet')
# initialize trainer
trainer = Trainer(model, learning_rate=learning_rate, epochs=epochs, evalEpoch=1)
# train the model
trainer.fit(
    trainDataLoader=trainDataLoader,
    testDataLoader=testDataLoader,
)
# save train logs and weights
np.save('./Results/history_CE_{}.npy'.format(labelfile[:-5]), trainer.logger)
torch.save(trainer.model, './Results/model_{}.pt'.format(labelfile[:-5]))