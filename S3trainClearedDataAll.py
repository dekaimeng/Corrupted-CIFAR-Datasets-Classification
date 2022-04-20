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


datasets = ['cifar10', 'cifar10', 'cifar10', 'cifar100', 'cifar100']
tasks = ['task1', 'task2', 'task3', 'task1', 'task2']
nClasses = [10, 10, 10, 100, 100]
datas = ['cifar-10-batches-py', 'cifar-10-batches-py', 'cifar-10-batches-py', 'cifar-100-python', 'cifar-100-python']
#%%
for dataset, task, nClass, data in zip(datasets, tasks, nClasses, datas):
    dataDir = os.path.join('Datasets', data)
    labelfile = '{}_noisy_labels_{}.json'.format(dataset, task)
    noisyLabelFile = os.path.join('.', 'Datasets', '{}_noisy_labels_{}.json'.format(dataset, task))
    noiseLabelArray = np.array(json.load(open(noisyLabelFile, "r")))

    epochs = 20  # train epochs
    batchSize = 512  # batch Size
    learning_rate = 5e-4  # learning rate

    #%% determinde noisy datas
    data = np.load(noisyLabelFile.replace('.json', '_noisy.npz'))
    X = data['X']
    y = data['y']
    probas = data['probas']
    # clearnlab to detect the noisy labels. Bool true is noisy samples
    noisyIndex = get_noise_indices(y, probas, prune_method='both', frac_noise=1, n_jobs=1)
    Xtrain = X[~noisyIndex, :]  # indice data that is not noisy
    ytrain = y[~noisyIndex]  # indice label that is not noisy
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
    # model = Classifier(nClass, inChannels=3, encoder_name='resnet18', pretrained='imagenet')
    model = Classifier(nClass, inChannels=3, encoder_name='densenet121', pretrained='imagenet')
    # model = Classifier(nClass, inChannels=3, encoder_name='timm-mobilenetv3_large_100', pretrained='imagenet')

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