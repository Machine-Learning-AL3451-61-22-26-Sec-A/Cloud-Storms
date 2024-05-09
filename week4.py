# Filename: neural_network_app.py
import streamlit as st
import numpy as np

class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o 

    def sigmoid(self, s):
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        return s * (1 - s)
    
    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error*self.sigmoidPrime(o)
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

# Scaling the data
def scale_data(X, y):
    X_scaled = X / np.amax(X, axis=0)
    y_scaled = y / 100
    return X_scaled, y_scaled

# Main function
def main():
    st.title('Neural Network with Streamlit')

    # Original data
    X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)     
    y = np.array(([92], [86], [89]), dtype=float)           

    # Scale the data
    X_scaled, y_scaled = scale_data(X, y)

    # Neural network initialization
    NN = Neural_Network()

    # Display original and scaled data
    st.subheader('Original Data:')
    st.write('Input:')
    st.write(X)
    st.write('Actual Output:')
    st.write(y)

    st.subheader('Scaled Data:')
    st.write('Scaled Input:')
    st.write(X_scaled)
    st.write('Scaled Output:')
    st.write(y_scaled)

    # Training the neural network
    NN.train(X_scaled, y_scaled)

    # Display predicted output and loss
    st.subheader('Prediction and Loss:')
    predicted_output = NN.forward(X_scaled)
    loss = np.mean(np.square(y_scaled - predicted_output))
    st.write('Predicted Output:')
    st.write(predicted_output)
    st.write('Loss:')
    st.write(loss)

    # Output the final weights
    st.subheader('Final Weights:')
    st.write('Weights from input to hidden layer:')
    st.write(NN.W1)
    st.write('Weights from hidden to output layer:')
    st.write(NN.W2)

if __name__ == "__main__":
    main()
