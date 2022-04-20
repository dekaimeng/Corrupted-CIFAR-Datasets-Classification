import numpy as np
from PIL import Image
import json
import os
from loader_for_CIFAR import *
from network import *
from trainers import *
#%% config
dataset = 'cifar10'
nClass = 10
dataDir = os.path.join('Datasets', 'cifar-10-batches-py')
noisyLabelFile = os.path.join('.', 'Datasets', '{}_noisy_labels_task1.json'.format(dataset))

epochs = 10  # train epochs
batchSize = 128  # batch Size
learning_rate = 1e-3  # learning rate
#%% load data
loader = cifar_dataloader(dataset, batchSize, 0, dataDir, noisyLabelFile)
trainDataLoader, noisyLabel, cleanLabel = loader.run('train')
testDataLoader = loader.run('test')

#%%
# initialize model
model = Classifier(nClass, inChannels=3, encoder_name='resnet18', pretrained='imagenet')
# initialize trainer
trainer = Trainer(model, learning_rate=learning_rate, epochs=epochs, evalEpoch=1)
# train the model
trainer.fit(trainDataLoader=trainDataLoader, testDataLoader=testDataLoader)
