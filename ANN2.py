import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


class ANN:
    def __init__(self, noOfInputUnits, noHiddenUnits, noOfOutputUnits, activationFunction):
        self.noOfInputUnits = noOfInputUnits
        self.noHiddenUnits = noHiddenUnits
        self.noOfOutputUnits = noOfOutputUnits
        self.activationFunction = activationFunction
        # initializing weights & biases with random values
        self.weightsInputHidden = np.random.uniform(low=0, high=1,
                                                      size=(self.noOfInputUnits, self.noHiddenUnits))

        self.weightsHiddenOutput = np.random.uniform(low=0, high=1,
                                                   size=(self.noHiddenUnits, self.noOfOutputUnits))

        self.biasHidden = np.random.uniform(low=0, high=1,
                                             size=(1, self.noHiddenUnits))

        self.biasOutput = np.random.uniform(low=0, high=1,
                                             size=(1, self.noOfOutputUnits))

    def pre_process(self, data):
        # df = pd.read_csv('student-mat.csv')
        data.dropna(inplace=True)
        labelenc = LabelEncoder()
        # data['Class'] = df['Class'].replace({'Kecimen': 0, 'Besni': 1})
        data.iloc[:,-1]=labelenc.fit_transform(data.iloc[:,-1])
        features = data.iloc[:, 0:7]
        target = data.iloc[:, 7:8]
        return features, target

    def split_data(self, data, testsize=0.2, randstate=39):
        features, target = self.pre_process(data)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        featuresTrain, featuresTest, targetTrain, targetTest = train_test_split(features, target, test_size=testsize, random_state=randstate)
        # labelenc = LabelEncoder()
        # y_train = labelenc.fit_transform(y_train)
        # y_test = labelenc.fit_transform(y_test)
        return featuresTrain, featuresTest, targetTrain, targetTest

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)


    def forward(self, inputs):
        if self.activationFunction == 'tanh':
            actFunction = self.tanh
        elif self.activationFunction == 'relu':
            actFunction = self.relu
        else:
            actFunction = self.sigmoid

        # net to hidden layer forward pass
        self.hiddenNet = np.dot(inputs, self.weightsInputHidden) + self.biasHidden
        self.hiddenOutput = actFunction(self.hiddenNet)

        # hidden to output layer forward pass
        self.outputInput = np.dot(self.hiddenOutput, self.weightsHiddenOutput) + self.biasOutput
        self.output = actFunction(self.outputInput)

        return self.output

    def train(self, featuresTrain, targetTrain, learning_rate=0.01, epochs=1000, batch_size=32):
        print(featuresTrain)
        print(targetTrain)

        for epoch in range(epochs):

            # shuffling the training data for each epoch to prevent the model from learning the order
            # shuffled_indices = np.random.permutation(len(featuresTrain))
            # featuresTrainShuffled = featuresTrain[shuffled_indices]
            # targetTrainShuffled = targetTrain[shuffled_indices]

            for i in range(1, len(featuresTrain), batch_size):
                featureBatch = featuresTrain[i:i + batch_size]
                targetBatch = targetTrain[i:i + batch_size]


                # Forward propagation
                predicted_output = self.forward(featureBatch)

                # Backpropagation
                error = targetBatch - predicted_output

                if self.activationFunction == 'tanh':
                    deltaOutput = error * (1 - np.square(predicted_output)**2)
                elif self.activationFunction == 'relu':
                    deltaOutput = np.where(predicted_output > 0, error, 0)
                else:
                    deltaOutput = error * predicted_output * (1 - predicted_output)

                # Update weights and biases for the output layer
                self.weightsHiddenOutput += self.hiddenOutput.T.dot(deltaOutput) * learning_rate
                self.biasOutput += np.sum(deltaOutput, axis=0)[0] * learning_rate

                # Calculate error_hidden and d_hidden for the hidden layer
                error_hidden = deltaOutput.dot(self.weightsHiddenOutput.T)
                if self.activationFunction == 'tanh':
                    deltaHidden = error_hidden * (1 - np.square(self.hiddenOutput))
                elif self.activationFunction == 'relu':
                    deltaHidden = np.where(self.hiddenOutput > 0, error_hidden, 0)
                else:
                    deltaHidden = error_hidden * self.hiddenOutput * (1 - self.hiddenOutput)

                # Update weights and biases for the hidden layer
                self.weightsInputHidden += featureBatch.T.dot(deltaHidden) * learning_rate
                self.biasHidden += np.sum(deltaHidden, axis=0)[0] * learning_rate

    def test(self, featuresTest):
        return self.forward(featuresTest)
if __name__ == "__main__":
    column_names = ["Area", "MajorAxisLength",	"MinorAxisLength",	"Eccentricity", "ConvexArea", "Extent", "Perimeter", "Class"]
    df = pd.read_excel('Raisin_Dataset.xlsx',names=column_names)
    input_values = pd.read_csv('input_values.csv')
    for index, row in input_values.iterrows():
        noOfHiddenUnits = int(row['hidden_size'])
        noOfOutputUnits = 1

        # act = input("Enter activation function to be used, available -- sigmoid, tanh, relu: ")
        act = row['activation'].lower()
        nn_model = ANN(7,noOfHiddenUnits,noOfOutputUnits,act)
        # data = nn_model.pre_process(df)
        featuresTrain, featuresTest, targetTrain, targetTest = nn_model.split_data(df)
        nn_model.train(featuresTrain, targetTrain)

        # Test the model
        train_predictions=nn_model.test(featuresTrain)

        test_predictions = nn_model.test(featuresTest)

        # Convert predicted probabilities to class labels
        train_predicted_labels = np.argmax(train_predictions, axis=1)
        test_predicted_labels = np.argmax(test_predictions, axis=1)

        # Calculate accuracy
        accuracy = accuracy_score(targetTrain, train_predicted_labels)
        print(f"Accuracy On Training Data: {accuracy * 100:.2f}%")
        # input_values.at[index, 'train_Accuracy'] = round(accuracy * 100,2)

        accuracy = accuracy_score(targetTest, test_predicted_labels)
        print(f"Accuracy On Test Data: {accuracy * 100:.2f}%")
        # input_values.at[index, 'test_Accuracy'] = round(accuracy * 100,2)

        # print(nn_model.forward(X.iloc[0]))