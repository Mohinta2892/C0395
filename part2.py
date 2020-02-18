import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.autograd import Variable

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class ClaimClassifier(nn.Module):

    def __init__(self, X_train, hidden):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        super(ClaimClassifier,self).__init__()

        self.n_input_dim = X_train.shape[1]

        self.n_hidden = hidden # Number of hidden nodes
        self.n_output = 1 # Number of output nodes = for binary classifier

        self.net = nn.Sequential(nn.Linear(self.n_input_dim, self.n_hidden),
                                 nn.ReLU(),
                                 nn.Linear(self.n_hidden, self.n_output),
                                 nn.Sigmoid())


    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A clean data set that is used for training and prediction.
        """
        # YOUR CODE HERE
        self.X=X_raw.to_numpy()

        self.maximum_x = np.max(self.X,axis=0)
        self.minimum_x = np.min(self.X,axis=0)
        self.X=(self.X-self.minimum_x) / (self.maximum_x -self.minimum_x)


        return self.X


    def fit(self, X_raw, y_raw, learning_rate=0.01, iters=1000):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)
        X_train=X_clean
        Y_train=y_raw

        # YOUR CODE HERE
        loss_func = nn.BCELoss()
        #learning_rate = 0.01
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)



        train_loss = []
        train_accuracy = []
        #iters = 1000
        Y_train_t = torch.FloatTensor(Y_train).reshape(-1, 1)
        for i in range(iters):
           X_train_t = torch.FloatTensor(X_train)
           y_hat = self.net(X_train_t)
           loss = loss_func(y_hat, Y_train_t)
           loss.backward()
           optimizer.step()
           optimizer.zero_grad()
           y_hat_class = np.where(y_hat.detach().numpy()<0.5, 0, 1)
           accuracy = np.sum(Y_train.reshape(-1,1)==y_hat_class) / len(Y_train)
           train_accuracy.append(accuracy)
           train_loss.append(loss.item())

    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)

        # YOUR CODE HERE
        X_test_t = torch.FloatTensor(X_clean)
        y_hat_test = self.net(X_test_t)
        #print(y_hat_test)
        y_hat_test_class = np.where(y_hat_test.detach().numpy()<0.5, 0, 1)
        #print(y_hat_test_class)
        return y_hat_test_class

        #return  # YOUR PREDICTED CLASS LABELS

    def evaluate_architecture(self, x, y):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        y_hat_test_class= self.predict(x)
        test_accuracy = np.sum(y.reshape(-1,1)==y_hat_test_class) / len(y)
        #print("Test Accuracy {:.2f}".format(test_accuracy))

        return test_accuracy


    def save_model(self):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model

# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch(x_train,y_train,x_val, y_val):
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """

    hidden = [10,20,30]
    learning_rate =[0.001,0.01, 1]
    iters= [1000, 2000, 3000]
    test_accuracy=0
    best_hyper_params=[]

    for i in range(3):
        net=ClaimClassifier(x_train, hidden[i])
        net.fit(x_train, y_train, learning_rate[i], iters[i])
        accur= net.evaluate_architecture(x_val,y_val)
          
        if accur> test_accuracy:
           test_accuracy=accur
           best_hyper_params.append({'hidden':hidden[i],'learning_rate':learning_rate[i], 'iters':iters[i]})

             


    return best_hyper_params # Return the chosen hyper parameters

def example_main():
    df=pd.read_csv('/home/samia/Downloads/neural_networks_35-master/part2_training_data.csv')

    y = df.iloc[1:,10]
    X = df.iloc[1:,:9]

    y=y.to_numpy()

    split_idx = int(0.8 * len(X))

    x_train = X[:split_idx]
    y_train = y[:split_idx]
    x_val = X[split_idx:]
    y_val = y[split_idx:]



    net = ClaimClassifier(x_train, 20)
    net.fit(x_train, y_train)
    test_accuracy= net.evaluate_architecture(x_val,y_val)

    print("Test Accuracy {:.2f}".format(test_accuracy))

    best_hyper_params=ClaimClassifierHyperParameterSearch(x_train,y_train, x_val,y_val)
    print(best_hyper_params)



if __name__ == "__main__":
    example_main()
