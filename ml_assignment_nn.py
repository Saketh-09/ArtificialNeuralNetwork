import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


class CustomNeuralNetwork:
    def __init__(self, input_size, hidden_layer_size, output_size, activation_function='sigmoid'):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.activation_function = activation_function

        # Initialize weights and biases based on random initialization in the specified range
        min_value = -5
        max_value = 5
        self.weights_input_hidden = np.random.uniform(low=min_value, high=max_value,
                                                      size=(self.input_size, self.hidden_layer_size))
        self.weights_hidden_output = np.random.uniform(low=min_value, high=max_value,
                                                       size=(self.hidden_layer_size, self.output_size))

        min_value = -4
        max_value = 4
        self.bias_hidden = np.random.uniform(low=min_value, high=max_value,
                                             size=(1, self.hidden_layer_size))
        self.bias_output = np.random.uniform(low=min_value, high=max_value,
                                             size=(1, self.output_size))

    # Define sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Define tanh function
    def tanh(self, x):
        return np.tanh(x)

    # Define ReLu function
    def relu(self, x):
        return np.maximum(0, x)

    # Define softmax function
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    # A method to do the forward pass
    def forward(self, inputs):
        # set the activation function
        if self.activation_function == 'sigmoid':
            activation_function = self.sigmoid
        elif self.activation_function == 'tanh':
            activation_function = self.tanh
        elif self.activation_function == 'relu':
            activation_function = self.relu

        # Forward pass through the network
        # Input to hidden layer
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = activation_function(self.hidden_input)

        # Hidden to output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        activation_function = self.softmax
        self.predicted_output = activation_function(self.output_input)

        return self.predicted_output

    # A method to do the training of the model with given data and set of hyperparameters
    def train(self, X_train, y_train, learning_rate=0.01, epochs=1000, batch_size=30):
        # Training the neural network
        for epoch in range(epochs):
            # Shuffle training data
            shuffled_indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[shuffled_indices]
            y_train_shuffled = y_train[shuffled_indices]

            for i in range(0, len(X_train_shuffled), batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]

                # Forward pass
                predicted_output = self.forward(X_batch)

                # backward pass
                error = y_batch - predicted_output

                if self.activation_function == 'sigmoid':
                    d_output = error * predicted_output * (1 - predicted_output)
                elif self.activation_function == 'tanh':
                    d_output = error * (1 - np.square(predicted_output) ** 2)
                elif self.activation_function == 'relu':
                    d_output = np.where(predicted_output > 0, error, 0)

                # Update the weights and the biases for the final layer after backward pass
                self.weights_hidden_output += self.hidden_output.T.dot(d_output) * learning_rate
                self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

                # Find error_hidden and d_hidden for the hidden layer
                error_hidden = d_output.dot(self.weights_hidden_output.T)
                if self.activation_function == 'sigmoid':
                    d_hidden = error_hidden * self.hidden_output * (1 - self.hidden_output)
                elif self.activation_function == 'tanh':
                    d_hidden = error_hidden * (1 - np.square(self.hidden_output))
                elif self.activation_function == 'relu':
                    d_hidden = np.where(self.hidden_output > 0, error_hidden, 0)

                # Update the weights and the biases for the hidden layer
                self.weights_input_hidden += X_batch.T.dot(d_hidden) * learning_rate
                self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    def test(self, X_test):
        # Testing the neural network
        return self.forward(X_test)


# A class for data preprocessing
class DataPreprocessor:
    def __init__(self):
        pass

    # Method to preprocess the data
    def preprocess_data(self, data):
        # Print the dataset
        print("The data in the given dataset is: ")
        print(data.to_string())
        print("\n")

        # Print summary statistics of the dataset
        print("Summary of this dataset's features is as following: ")
        print(data.describe())
        print("\n")

        # Encode the dependent feature
        label_encoder = LabelEncoder()
        data.iloc[:, -1] = label_encoder.fit_transform(data.iloc[:, -1])
        print("The dependent feature Y encoded is as follows : ")
        print(data.iloc[:, -1])
        print("\n")

        # Extract features and target variables
        X = data.iloc[:, 0:7]
        y = data.iloc[:, 7]
        return X, y

    # Method for splitting the data into training and testing data sets
    def split_the_data(self, X, y, testsize=0.2, randstate=39):
        # Standardize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=randstate)

        # Encode the target variables
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.fit_transform(y_test)

        return X_train, y_train, X_test, y_test


# Read the data from CSV file
raw_data = pd.read_csv("https://raw.githubusercontent.com/Saketh-09/ArtificialNeuralNetwork/master/Raisin_Dataset.csv")

# Initialize the DataPreprocessor object
data_preprocessor = DataPreprocessor()

# Preprocess the data
X, y = data_preprocessor.preprocess_data(raw_data)

result_for_variation_of_hyper_parameters = pd.read_csv('hyperparameters.csv')
test_size = 25
test_size = test_size / 100.0

random_state = np.random.randint(0, 1000)

# Split the data into training and testing sets
X_train, y_train, X_test, y_test = data_preprocessor.split_the_data(X, y, test_size, random_state)

# Print information about the split data
print("Training data shape:")
print(X_train.shape, y_train.shape)
print("\n")

print("Testing data shape:")
print(X_test.shape, y_test.shape)

input_size = X_train.shape[1]
print("Total number of neurons in the input layer are as follows:", input_size)

"""
Iterate through each row in the csv file which will allow us to train and test our model for different set of
 hyperparameter values
"""
for i, r in result_for_variation_of_hyper_parameters.iterrows():
    # Extract hyperparameter values from the row
    hidden_layer_size = int(r['hidden_layer_size'])
    output_layer_size = 2

    # Choose activation function based on the row's 'activation' value
    # activation function -- sigmoid, tanh, relu: ")
    activation_function = r['activation_function'].lower()
    neural_network = CustomNeuralNetwork(input_size, hidden_layer_size, output_layer_size, activation_function=activation_function)

    # Transforming output labels to one-hot encoding
    y_train_one_hot = np.eye(output_layer_size)[y_train]

    # Training the model
    # alpha is learning rate
    # ep is number of epochs
    # bt is our batch size
    alpha = float(r['learning_rate'])
    ep = int(r['epochs'])
    bt = int(r['batch_size'])
    neural_network.train(X_train, y_train_one_hot, learning_rate=alpha, epochs=ep, batch_size=bt)

    # Test the model
    train_predictions = neural_network.test(X_train)

    test_predictions = neural_network.test(X_test)

    # Convert predicted probabilities to class labels
    train_predicted_labels = np.argmax(train_predictions, axis=1)
    test_predicted_labels = np.argmax(test_predictions, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_train, train_predicted_labels)
    print(f"Accuracy On Training Data: {accuracy * 100:.2f}%")
    result_for_variation_of_hyper_parameters.at[i, 'train_data_accuracy'] = round(accuracy * 100, 3)

    accuracy = accuracy_score(y_test, test_predicted_labels)
    print(f"Accuracy On Test Data: {accuracy * 100:.2f}%")
    result_for_variation_of_hyper_parameters.at[i, 'test_data_accuracy'] = round(accuracy * 100, 3)
result_for_variation_of_hyper_parameters.to_csv("results.csv", index=False)