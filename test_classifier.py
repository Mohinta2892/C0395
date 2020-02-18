import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd


df=pd.read_csv('/home/samia/Downloads/neural_networks_35-master/part2_training_data.csv')

y = df.iloc[1:,10]
X = df.iloc[1:,:9]

#print(X)

X=X.to_numpy()
y=y.to_numpy()

maximum_x = np.max(X,axis=0)
minimum_x = np.min(X,axis=0)
X=(X-minimum_x) / (maximum_x -minimum_x)

#X_train, X_test, Y_train, Y_test = train_test_split(X, y,
#    test_size=0.3, random_state=73)

split_idx = int(0.7 * len(X))

X_train = X[:split_idx]
Y_train = y[:split_idx]
X_test = X[split_idx:]
Y_test = y[split_idx:]

#x_train = Variable(torch.from_numpy(x_train))
#y_train = Variable(torch.from_numpy(y_train))

#X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
#Y_train = torch.from_numpy(Y_train).type(torch.LongTensor)
#X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
#Y_test = torch.from_numpy(Y_test).type(torch.LongTensor)


# Define network dimensions
n_input_dim = X_train.shape[1]
# Layer size
n_hidden = 10 # Number of hidden nodes
n_output = 1 # Number of output nodes = for binary classifier

# Build your network
net = nn.Sequential(
    nn.Linear(n_input_dim, n_hidden),
    nn.ELU(),
    nn.Linear(n_hidden, n_output),
    nn.Sigmoid())
    
print(net)


loss_func = nn.BCELoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


train_loss = []
train_accuracy = []
iters = 1000
Y_train_t = torch.FloatTensor(Y_train).reshape(-1, 1)
for i in range(iters):
    X_train_t = torch.FloatTensor(X_train)
    y_hat = net(X_train_t)
    loss = loss_func(y_hat, Y_train_t)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    y_hat_class = np.where(y_hat.detach().numpy()<0.5, 0, 1)
    accuracy = np.sum(Y_train.reshape(-1,1)==y_hat_class) / len(Y_train)
    train_accuracy.append(accuracy)
    train_loss.append(loss.item())

'''    
fig, ax = plt.subplots(2, 1, figsize=(12,8))
ax[0].plot(train_loss)
ax[0].set_ylabel('Loss')
ax[0].set_title('Training Loss')

ax[1].plot(train_accuracy)
ax[1].set_ylabel('Classification Accuracy')
ax[1].set_title('Training Accuracy')

plt.tight_layout()
plt.show()'''


# Pass test data
X_test_t = torch.FloatTensor(X_test)
y_hat_test = net(X_test_t)
y_hat_test_class = np.where(y_hat_test.detach().numpy()<0.5, 0, 1)
test_accuracy = np.sum(Y_test.reshape(-1,1)==y_hat_test_class) / len(Y_test)
print("Test Accuracy {:.2f}".format(test_accuracy))



import sklearn as sk
from sklearn.neural_network import MLPClassifier

NN = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(10, 2), random_state=1)
NN.fit(X_train, Y_train)
NN.predict(X_test)
print(round(NN.score(X_train,Y_train), 4))

