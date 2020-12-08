# Databricks notebook source
# load packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# COMMAND ----------

# generate random data
np.random.seed(123)
features = np.random.uniform(0, 1, size=(100, 2))
target = np.random.randint(2, size=100).reshape(100,)

# COMMAND ----------

# shuffle data
idx = np.arange(target.shape[0])
np.random.shuffle(idx)

# train, test
X_test, y_test = features[idx[:25]], target[idx[:25]]
X_train, y_train = features[idx[25:]], target[idx[25:]]

# normalize
mu, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
X_train, X_test = (X_train - mu) / std, (X_test - mu) / std

# COMMAND ----------

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
# manual logistic regression
class LogisticRegression():
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = torch.zeros(1, num_features, 
                                   dtype=torch.float32, device=device)
        self.bias = torch.zeros(1, dtype=torch.float32, device=device)

        # forward propagation
    def forward(self, x):
        linear = torch.add(torch.mm(x, self.weights.t()), self.bias).view(-1)
        probas = self._sigmoid(linear)
        return probas
      
        # backward propagation
    def backward(self, x, y, probas):  
        grad_loss_out = y - probas.view(-1)
        grad_loss_w = -torch.mm(x.t(), grad_loss_out.view(-1, 1)).t()
        grad_loss_b = -torch.sum(grad_loss_out)
        return grad_loss_w, grad_loss_b
      
        # activation function
    def _sigmoid(self, z):
        return 1. / (1. + torch.exp(-z))
    
        # loss function: binary cross entropy/negative log-likelihood
    def _logit_cost(self, y, proba):
        tmp1 = torch.mm(-y.view(1, -1), torch.log(proba.view(-1, 1)))
        tmp2 = torch.mm((1 - y).view(1, -1), torch.log(1 - proba.view(-1, 1)))
        return tmp1 - tmp2
    
    def train(self, x, y, num_epochs, learning_rate=0.01):
        epoch_cost = []
        for e in range(num_epochs):
            
            #### Compute outputs ####
            probas = self.forward(x)
            preds = torch.where(probas > 0.5, torch.tensor(1, device=device), torch.tensor(0, device=device)).view(-1)
            correct = preds.eq(y).sum().item()
            
            #### Compute gradients ####
            grad_w, grad_b = self.backward(x, y, probas)

            #### Update weights ####
            self.weights -= learning_rate * grad_w
            self.bias -= learning_rate * grad_b
            
            #### Logging ####
            cost = self._logit_cost(y, self.forward(x)) / x.size(0)
            print('Epoch: %03d' % (e+1), end="")
            print(' | Train ACC: %.3f' % correct, end="")
            print(' | Cost: %.3f' % cost)
            epoch_cost.append(cost)
        return epoch_cost

# COMMAND ----------

# turn numpy to tensors
X_train_tensor = torch.as_tensor(X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32, device=device)

# instantiate the model
model = LogisticRegression(num_features=2)
# run the model
epoch_cost = model.train(X_train_tensor, y_train_tensor, num_epochs=15, learning_rate=0.1)

print('\nModel parameters:')
print('  Weights: %s' % model.weights)
print('  Bias: %s' % model.bias)

# COMMAND ----------


