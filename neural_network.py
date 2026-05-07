import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')
data = np.array(data)
m,n = data.shape
np.random.shuffle(data)


data_dev = data[0:1000].T #It use because we want to use the first 1000 data for dev set and here "dev" means development set in in hindi word we call it "vikas set"
Y_dev = data_dev[0] #Here we are taking the first row of the dev set as the labels because in the given dataset the first column is the label column
X_dev = data_dev[1:n] #Here we are taking the rest of the rows as the features because in the given dataset the first column is the label column and the rest of the columns are the feature columns
X_dev = X_dev/255.0


data_train = data[1000:m].T #It use because we want to use the rest of the data for training set and here "train" means training set in in hindi word we call it "prashikshan set"
Y_train = data_train[0] #Here we are taking the first row of the training set as the labels because in the given dataset the first column is the label column
X_train = data_train[1:n] #Here we are taking the rest of the rows as the features because in the given dataset the first column is the label column and the rest of the columns are the feature columns
X_train = X_train/255.0
_,mtrain = X_train.shape

# print(data_train)
# print()
# print(Y_train)
# print()
# print(X_train)   


def intial_W_b():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1,b1,W2,b2

def ReLU(X):
    return np.maximum(0,X)

def softmax(X):
    exp_X = np.exp(X - np.max(X, axis=0, keepdims=True))
    return exp_X / np.sum(exp_X, axis=0, keepdims=True)

def froward_propagation(X,W1,b1,W2,b2):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1,A1,Z2,A2

def ReLU_derivative(x):
    return x > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size),Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_propagation(Z1,A1,Z2,A2,W1,W2,X,Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_derivative(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1,db1,dW2,db2

def update_parameters(W1, b1,W2,b2,dW1,db1,dW2,db2,alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1,b1,W2,b2

def get_predictions(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions,Y):
    print(predictions,Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X,Y,alpha, iterations):
    W1,b1,W2,b2 = intial_W_b()
    for i in range(iterations):
        Z1,A1,Z2,A2 = froward_propagation(X,W1,b1,W2,b2)
        dW1,db1,dW2,db2 = backward_propagation(Z1,A1,Z2,A2,W1,W2,X,Y)
        W1,b1,W2,b2 = update_parameters(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha)
        if i % 1200 == 0:
            print("Iteration: ",i)
            predictions = get_predictions(A2)
            print("Accuracy: ",get_accuracy(predictions,Y))
    
    return W1,b1,W2,b2, predictions


W1,b1,W2,b2,predictions = gradient_descent(X_train,Y_train,0.1,5500)

print("Train set accuracy:",get_accuracy(predictions,Y_train)*100, "%")


answer = "y"
while answer == "y":
    img_path = input("Enter Image Path: ")
    img = Image.open(img_path).convert('L')
    img = img.resize((28,28))
    img_array = np.array(img)
    img_array = img_array.reshape(784,1)
    img_array = img_array/255.0
    Z1,A1,Z2,A2 = froward_propagation(img_array,W1,b1,W2,b2)
    predicted_label = get_predictions(A2)
    print("Predicted Label: ", predicted_label[0])
    answer = input("Do you want to predict another image? (y/n): ")




