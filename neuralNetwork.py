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

        self.wInputHidden = numpy.random.rand(self.hnodes, self.inodes) -0.5
        self.wHiddenOutput = numpy.random.rand(self.onodes, self.hnodes) -0.5

        self.activation_function = lambda x: scipy.special.expit(x)


    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wInputHidden, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.wHiddenOutput, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.wHiddenOutput.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.wHiddenOutput += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wInputHidden += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
    

    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wInputHidden, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.wHiddenOutput, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


neuralNetwork = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
input_list =[(1.0, 0.5, 0.3), (0.4, 0.9, 0.1)]
target_list =[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

neuralNetwork.train(input_list, target_list)

# test query (doesn't mean anything useful yet)
output = neuralNetwork.query([1.0, 0.5, -1.5])
print("Queryoutput:")
print(output)