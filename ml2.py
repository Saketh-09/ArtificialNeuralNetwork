# -*- coding: utf-8 -*-
"""ML2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qfvljZ4dbaq--mH_ZiyFd-SKDyq03mO4
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation_function='sigmoid'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_function = activation_function

        # c=int(input("Enter 1 to enter weights and bias manually/ enter -1 for initializing weights randomly and initializing all biases 0  "))
        c=-1
        if(c==1):
          # Initialize weights and biases
          self.weights_input_hidden = self.get_user_input_values(
              input_size, hidden_size, "Enter weights for input to hidden layer (comma-separated): "+str(input_size*hidden_size)+" values  ")
          self.bias_hidden = self.get_user_input_values(
              1, hidden_size, "Enter biases for hidden layer (comma-separated): "+str(hidden_size)+" values  ")
          self.weights_hidden_output = self.get_user_input_values(
              hidden_size, output_size, "Enter weights for hidden to output layer (comma-separated): "+str(hidden_size*output_size)+" values  ")
          self.bias_output = self.get_user_input_values(
              1, output_size, "Enter biases for output layer (comma-separated): "+str(output_size)+" values  ")

        else:
          # Define the range for random weight initialization
          print("\n\nTip! initializing Weights between 0 and 1 will help in developing better model for this dataset ")
          # min_value = int(input("Enter the Min value of the weights to initialize  "))
          # max_value = int(input("Enter the Max value of the weights to initialize  "))
          min_value = -5
          max_value = 5
          # Initialize weights with random values in the specified range
          self.weights_input_hidden = np.random.uniform(low=min_value, high=max_value,
                                                        size=(self.input_size, self.hidden_size))

          self.weights_hidden_output = np.random.uniform(low=min_value, high=max_value,
                                                         size=(self.hidden_size, self.output_size))

          min_value = -4
          max_value = 4
          self.bias_hidden = np.random.uniform(low=min_value, high=max_value,
                                                        size=(1, self.hidden_size))

          self.bias_output = np.random.uniform(low=min_value, high=max_value,
                                                        size=(1, self.output_size))



    def get_user_input_values(self, input_size, output_size, prompt):
        print(prompt)
        values = input().strip().split(',')
        if len(values) != (input_size * output_size):
            raise ValueError(f"Invalid number of values. Expected {input_size * output_size} values.")
        return np.array(values, dtype=float).reshape((input_size, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)



    def forward(self, inputs):
        if self.activation_function == 'sigmoid':
            activation_function = self.sigmoid
        elif self.activation_function == 'tanh':
            activation_function = self.tanh
        elif self.activation_function == 'relu':
            activation_function = self.relu

        # Input to hidden layer
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = activation_function(self.hidden_input)

        # Hidden to output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        activation_function = self.softmax
        self.predicted_output = activation_function(self.output_input)

        return self.predicted_output

    def train(self, X_train, y_train, learning_rate=0.01, epochs=1000, batch_size=32):
        for epoch in range(epochs):

            # Shuffle the training data for each epoch
            shuffled_indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[shuffled_indices]
            y_train_shuffled = y_train[shuffled_indices]

            for i in range(0, len(X_train_shuffled), batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]


                # Forward propagation
                predicted_output = self.forward(X_batch)

                # Backpropagation
                error = y_batch - predicted_output

                if self.activation_function == 'sigmoid':
                    d_output = error * predicted_output * (1 - predicted_output)
                elif self.activation_function == 'tanh':
                    d_output = error * (1 - np.square(predicted_output)**2)
                elif self.activation_function == 'relu':
                    d_output = np.where(predicted_output > 0, error, 0)

                # Update weights and biases for the output layer
                self.weights_hidden_output += self.hidden_output.T.dot(d_output) * learning_rate
                self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

                # Calculate error_hidden and d_hidden for the hidden layer
                error_hidden = d_output.dot(self.weights_hidden_output.T)
                if self.activation_function == 'sigmoid':
                    d_hidden = error_hidden * self.hidden_output * (1 - self.hidden_output)
                elif self.activation_function == 'tanh':
                    d_hidden = error_hidden * (1 - np.square(self.hidden_output))
                elif self.activation_function == 'relu':
                    d_hidden = np.where(self.hidden_output > 0, error_hidden, 0)

                # Update weights and biases for the hidden layer
                self.weights_input_hidden += X_batch.T.dot(d_hidden) * learning_rate
                self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate



    def test(self, X_test):
        return self.forward(X_test)

class preprocessing:
    def __init__(self):
        pass
    def preprocess_data(self, data):
        #removing rows having na values
        data=data.dropna()

        print("The dataset is: ")
        print(data.to_string())
        print("\n\n\n")
        print("Summary of dataset's features: ")
        print(data.describe())
        print("\n\n\n")

        labelenc = LabelEncoder()
        data.iloc[:,-1]=labelenc.fit_transform(data.iloc[:,-1])
        print("The dependent feature is encoded : ")
        print(data.iloc[:,-1])
        print("\n\n\n")

        unique_classes = data['Class_labels'].unique()
        custom_palette = sns.color_palette("husl", n_colors=len(unique_classes))

        # sns.pairplot(data,hue='Class_labels',palette=custom_palette)
        # print("The pairplot below is used to visualize whole dataset")
        # plt.show()
        # plt.close()


        X = data.iloc[:, 0:4]
        y = data.iloc[:, 4]
        return X,y



    def split_data(self,X,y,testsize=0.2,randstate=39):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=randstate)
        labelenc = LabelEncoder()
        y_train= labelenc.fit_transform(y_train)
        y_test= labelenc.fit_transform(y_test)

        return X_train, y_train, X_test, y_test

column_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']
data = pd.read_csv("https://raw.githubusercontent.com/KrishnaRohithVemulapalli/MLproject2/main/iris.data",names=column_names)
pp=preprocessing()

X,y=pp.preprocess_data(data)

input_values = pd.read_csv('input_values.csv')

test_size=30
test_size=test_size/100.0

random_state = np.random.randint(0, 1000)

X_train, y_train, X_test, y_test = pp.split_data(X, y, test_size, random_state)
input_size = X_train.shape[1]

print("Number of nuerons in input layer are :",input_size)
for index, row in input_values.iterrows():
    hidden_size = int(row['hidden_size'])
    output_size = 3  # For multi-class classification

    # act = input("Enter activation function to be used, available -- sigmoid, tanh, relu: ")
    act = row['activation'].lower()
    nn = NeuralNetwork(input_size, hidden_size, output_size, activation_function=act)

    # Convert target labels to one-hot encoding
    y_train_one_hot = np.eye(output_size)[y_train]
    # Train the model
    alpha=float(row['alpha'])
    ep=int(row['epoch'])
    bt=int(row['batch_size'])
    # print(act)
    # print(alpha)
    # print(bt)
    # print(hidden_size)
    # print(ep)
    nn.train(X_train, y_train_one_hot, learning_rate=alpha, epochs=ep,batch_size=bt)

    # Test the model
    train_predictions=nn.test(X_train)


    test_predictions = nn.test(X_test)

    # Convert predicted probabilities to class labels
    train_predicted_labels = np.argmax(train_predictions, axis=1)
    test_predicted_labels = np.argmax(test_predictions, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_train, train_predicted_labels)
    print(f"Accuracy On Training Data: {accuracy * 100:.2f}%")
    input_values.at[index, 'train_Accuracy'] = round(accuracy * 100,2)

    accuracy = accuracy_score(y_test, test_predicted_labels)
    print(f"Accuracy On Test Data: {accuracy * 100:.2f}%")
    input_values.at[index, 'test_Accuracy'] = round(accuracy * 100,2)
input_values.to_csv("input_values.csv", index=False)

