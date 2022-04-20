import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np


class Trainer():
    """general trainer
    """
    def __init__(self, model, criterion=None, learning_rate=2e-4, epochs=100, logCols=140, evalEpoch=5, evalEnable=True):
        self.learning_rate = learning_rate  # learning rate
        self.epochs = epochs  # epochs

        # Build Model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        # adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # loss function
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        # other
        self.logCols = logCols
        self.evalEpoch = evalEpoch
        self.evalEnable = evalEnable
        self.logger = {'trainloss': [], 'testloss': [], 'trainAcc': [], 'testAcc': []}

    def fit(self, trainDataLoader, testDataLoader):
        """training iteration

        Args:
            trainDataLoader (dataLoader): training data loader
            testDataLoader (dataLoader): testing data loader
        """
        for epoch in range(self.epochs + 1):
            self.model.train()
            pbar = tqdm(trainDataLoader, ncols=self.logCols)  # initialize pbar
            pbar.set_description('Epoch {:2d}'.format(epoch))

            # train period
            temp, NtrueTrain, NtrueReal, N = {}, 0, 0, 0
            for n_step, batch_data in enumerate(pbar):
                # get data
                xData = batch_data[0].float().to(self.device)
                yTrain = batch_data[1].long().to(self.device)
                # forward and backward
                yPred = self.model(xData)
                loss = self.criterion(yPred, yTrain)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # metric and log data
                yTrain = yTrain.cpu().data.numpy()
                yPred = yPred.cpu().data.numpy()
                yPred = np.argmax(yPred, axis=1)

                NtrueTrain += np.sum(yPred == yTrain)
                N += xData.shape[0]
                accuracyTrain = NtrueTrain / N

                temp['CE'] = loss.item()
                temp['AccTrain'] = accuracyTrain
                pbar.set_postfix(temp)
            self.logger['trainloss'].append(loss.item())
            self.logger['trainAcc'].append(accuracyTrain)

            if self.evalEnable and epoch % self.evalEpoch == 0:
                # test period
                print('*' * self.logCols)
                self.evaluation(testDataLoader)
                print('*' * self.logCols)

        return self

    def evaluation(self, testDataLoader):
        """evaluation the model

        Args:
            testDataLoader (dataLoader): testing data loader
        """
        self.model.eval() # set model mode
        pbar = tqdm(testDataLoader, ncols=self.logCols)
        pbar.set_description('Test')
        temp, Ntrue, N = {}, 0, 0

        for n_step, batch_data in enumerate(pbar):
            # get data
            xData = batch_data[0].float().to(self.device)
            yTrue = batch_data[1].long().to(self.device)
            # forward
            with torch.no_grad():
                yPred = self.model(xData)
            loss = self.criterion(yPred, yTrue)
            # metric and log result
            yTrue = yTrue.cpu().data.numpy()
            yPred = yPred.cpu().data.numpy()
            yPred = np.argmax(yPred, axis=1)
            Ntrue += np.sum(yPred == yTrue)
            N += xData.shape[0]
            accuracy = Ntrue / N
            temp['CE'] = loss.item()
            temp['Acc'] = accuracy
            pbar.set_postfix(temp)

        self.logger['testloss'].append(loss.item())
        self.logger['testAcc'].append(accuracy)

        return self