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

#%% config
datasets = ['cifar10', 'cifar10', 'cifar10', 'cifar100', 'cifar100']
tasks = ['task1', 'task2', 'task3', 'task1', 'task2']
nClasses = [10, 10, 10, 100, 100]
datas = ['cifar-10-batches-py', 'cifar-10-batches-py','cifar-10-batches-py','cifar-100-python','cifar-100-python']

for dataset, task, nClass, data in zip(datasets, tasks, nClasses, datas):
    # dataset = 'cifar10'
    # nClass = 10
    dataDir = os.path.join('Datasets', data)
    noisyLabelFile = os.path.join('.', 'Datasets', '{}_noisy_labels_{}.json'.format(dataset, task))
    noiseLabelArray = np.array(json.load(open(noisyLabelFile, "r")))

    epochs = 10 # train epochs
    batchSize = 512 # batch size
    learning_rate = 1e-3 # learning rate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #%% k folds data for cross validation
    print('Get all train data...')
    loader = cifar_dataloader(dataset, batchSize, 0, dataDir, noisyLabelFile) # initialize the data loader
    transforms_train = loader.transform_train # transforms
    transforms_test = loader.transform_test
    trainDataLoader, noisyLabel, cleanLabel = loader.run('train')
    train_data = trainDataLoader.dataset.train_data
    # collect all the train data and their labels for cross validation
    X, y = [], []
    for batch_data in trainDataLoader:
        index = batch_data[2]
        X.append(train_data[index])
        y.append(batch_data[1])
    X = np.concatenate(X, axis=0)
    y = np.hstack(y)

    # generate dataloaders for cross validation
    trainDataLoaders, validDataLoaders = [], []
    trainNoisyLabels = []
    kf = KFold(5) # K fold
    for trainIdx, validIdx in kf.split(X):
        # train datast depend on train Index
        dataset = simple_cifar_dataset(X[trainIdx], y[trainIdx], transform=transforms_train) 
        trainDataLoaders.append(DataLoader(dataset, batch_size=batchSize, shuffle=True))
        trainNoisyLabels.append(y[trainIdx])

        # validation dataset depend on validation index
        dataset = simple_cifar_dataset(X[validIdx], y[validIdx], transform=transforms_test)
        validDataLoaders.append(DataLoader(dataset, batch_size=batchSize, shuffle=False))

    #%% cross validtion train
    nSamples = len(y)

    Xall, yall, probas = [], [], []
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
            xData = batch_data[0].float().to(device) # load to device
            with torch.no_grad():
                pred = model(xData)
                pred = torch.softmax(pred, dim=-1)
            probas.append(pred.detach().cpu().numpy())
            Xall.append(batch_data[-1])
            yall.append(batch_data[1])

    Xall = np.concatenate(Xall, axis=0)
    yall = np.hstack(yall)
    probas = np.vstack(probas)
    # save all the result for next step
    np.savez(noisyLabelFile.replace('.json', '_noisy.npz'), X=Xall, y=yall, probas=probas)
