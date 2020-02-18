import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.autograd import Variable

class ClaimClassifier(nn.Module):

    def __init__(self):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        super(ClaimClassifier,self).__init__()

		
    #this has been implemented by me.
    def forward(self,x):
        #Output of the first layer
        x = self.fc1(x)
        #Activation function is Relu. Feel free to experiment with this
        x = F.tanh(x)
        #This produces output
        x = self.fc2(x)
        return x
		
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
        self.maximum_x = np.max(X_raw,axis=0) #axis = 0 across rows, covers all rows for each column
        self.minimum_x = np.min(X_raw,axis=0)
        return (X_raw-self.minimum_x) / (self.maximum_x -self.minimum_x)


    def fit(self, X_raw, y_raw):
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
        X_clean = X_raw #self._preprocessor(X_raw)
        # YOUR CODE HERE
        #Initialize the model        
        model = ClaimClassifier()
        #Define loss criterion
        criterion = nn.CrossEntropyLoss()
        #Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
		#Number of epochs
        epochs = 10000
        #List to store losses
        losses = []
        for i in range(epochs):
        #Predict the output for Given input
           y_pred = model.forward(X_clean.float())
           #Compute Cross entropy loss
           loss = criterion(y_pred,y_raw)
           #Add loss to the list
           losses.append(loss.item())
           #Clear the previous gradients
           optimizer.zero_grad()
           #Compute gradients
           loss.backward()
           #Adjust weights
           optimizer.step()

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
        X_clean = X_raw #self._preprocessor(X_raw)

        # YOUR CODE HERE
		#Apply softmax to output. 
        pred = F.softmax(self.forward(X_clean))
        ans = []
        #Pick the class with maximum weight
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

        #return  # YOUR PREDICTED CLASS LABELS

    def evaluate_architecture(self):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        #model = ClaimClassifier()
        #X_clean = self._preprocessor(X_raw)

        #print(accuracy_score(model.predict(X_clean),y))
        pass


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
def ClaimClassifierHyperParameterSearch():
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """
    pass
    #return  # Return the chosen hyper parameters

def example_main():
    df=pd.read_csv('/home/samia/Downloads/neural_networks_35-master/part2_training_data.csv')

    y = df.iloc[1:,10]
    X = df.iloc[1:,:9]

    X=X.to_numpy()
    y=y.to_numpy()

    split_idx = int(0.7 * len(X))

    x_train = X[:split_idx]
    y_train = y[:split_idx]
    x_val = X[split_idx:]
    y_val = y[split_idx:]

    #x_train = Variable(torch.from_numpy(x_train))
    #y_train = Variable(torch.from_numpy(y_train))

    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    x_val = torch.from_numpy(x_val).type(torch.FloatTensor)
    y_val = torch.from_numpy(y_val).type(torch.LongTensor)



    net = ClaimClassifier()
    net.fit(x_train, y_train)
    print(accuracy_score(net.predict(x_val),y_val))


if __name__ == "__main__":
    example_main()
