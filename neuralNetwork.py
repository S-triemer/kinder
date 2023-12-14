import numpy
import scipy.special
import matplotlib.pyplot

input_nodes = 3
output_nodes = 3
hidden_nodes = 3
learning_rate = 0.5

class neuralNetwork():

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

    def train(self):
        print("I'am training")
    

    def query(self):
        print("I ask...")


neuralNetwork = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
neuralNetwork.train()
neuralNetwork.query()
