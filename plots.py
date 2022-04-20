import numpy as np
import matplotlib.pyplot as plt
import os
#Cifar dataset graph:
datasets = ['cifar10', 'cifar10', 'cifar10', 'cifar100', 'cifar100']
tasks = ['task1', 'task2', 'task3', 'task1', 'task2']
nClasses = [10, 10, 10, 100, 100]
datas = ['cifar-10-batches-py', 'cifar-10-batches-py', 'cifar-10-batches-py', 'cifar-100-python', 'cifar-100-python']
for dataset, task, nClass, data in zip(datasets, tasks, nClasses, datas):
    logger = np.load(os.path.join('Results', 'history_CE_{}_noisy_labels_{}.npy'.format(dataset, task)), allow_pickle=True).item()
    plt.figure(figsize=(10, 8))
    plt.plot(logger['trainAcc'], label='trainAcc')
    plt.plot(logger['testAcc'], label='testAcc')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid('on')
    plt.ylim([0, 1])
    plt.savefig(os.path.join('Results', 'history_{}_noisy_labels_{}.png'.format(dataset, task)))
#animal_10N dataset graph:
logger = np.load(os.path.join('Results', 'history_CE_animal.npy'), allow_pickle=True).item()
plt.figure(figsize=(10, 8))
plt.plot(logger['trainAcc'], label='trainAcc')
plt.plot(logger['testAcc'], label='testAcc')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid('on')
plt.ylim([0, 1])
plt.savefig(os.path.join('Results', 'history_CE_animal.npy.png'.format(dataset, task)))