import numpy as np
import cleanlab
from sklearn.mixture import GaussianMixture
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from cleanlab.pruning import get_noise_indices

plt.close('all')

#%% Generate data
nSamples = 500
X, y = datasets.make_blobs(n_samples=nSamples, random_state=8) 
nClass = len(np.unique(y))
# plot the data
plt.figure()
for i in np.unique(y):
    plt.scatter(X[y==i, 0], X[y==i, 1])

#%% Noisy labels 
noisyIndex = np.random.randint(0, nSamples, 5) # random select samples to change label
yNoisy = y*1.0
yNoisy[noisyIndex] = (y[noisyIndex] - 1) % nClass # channel the true label
label = yNoisy.astype(np.int)

#%% unsupervied GMM with defined initial centers
# centers = np.zeros([nClass, X.shape[1]]) # initialize the cluster centers
# for i in range(nClass):
#     centers[i, :] = np.mean(X[label==i, :], axis=0)
    
# gmm = GaussianMixture(n_components=nClass, means_init=centers).fit(X)
# psx = gmm.predict_proba(X)

#%% K-fold classification
kfold = 5
kf = KFold(n_splits=kfold)

psx = np.zeros([nSamples, nClass])
for trainIndex, testIndex in kf.split(X):
    # train data/ valdation data
    xtrain = X[trainIndex, :]
    ytrain = y[trainIndex]
    xtest = X[testIndex, :]
    
    model = LogisticRegression().fit(xtrain, ytrain) # fit the model
    psx[testIndex, :] = model.predict_proba(xtest) # predict the probability


# s: noisy labels
# psx: the predicted probabilities of n x m, processed by cross-validation/unsupervised learning
baseline_cl_both  = get_noise_indices(label, psx, prune_method='both', frac_noise=1, n_jobs=1)
noisyIndexPred = np.argwhere(baseline_cl_both ).squeeze()
noisyIndex = np.argwhere(yNoisy != y).squeeze()
print('Real Noisy Index:', noisyIndex)
print('Pred Noisy Index:', noisyIndexPred)

# assert len(noisyIndex) == len(noisyIndexPred)
# assert (np.argwhere(yNoisy != y).squeeze() == noisyIndexPred).all()