import math
import random

class function:
    def linear(x):
        return (x if x >= 0 else 0)
    def tanh(x):
        return math.tanh(x)
    def sigmoid(x):
        return (1 / (1 + math.exp(-x)))

class function_inc:
    def linear(p):
        return (1 if p >= 0 else 0)
    def tanh(p):
        return 1 - (p ** 2)
    def sigmoid(p):
        return p * (1 - p)

class noron:
    def __init__(self , input_size , func):
        self.input_size = input_size
        self.weights = [random.uniform(-0.5 , 0.5) for i in range(input_size)]
        self.bias = random.uniform(-0.5 , 0.5)
        if func == 'linear':
            self.activation_function = function.linear
        elif func == 'tanh':
            self.activation_function = function.tanh
        elif func == 'sigmoid':
            self.activation_function = function.sigmoid
        else:
            raise ValueError("Invalid activation function type. Choose from 'linear', 'tanh', or 'sigmoid'.")
        

    def forward(self , inputs):
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, but got {len(inputs)}.")
        
        weighted_sum = sum(w * i for w, i in zip(self.weights , inputs)) + self.bias
        return self.activation_function(weighted_sum)

class layer:
    def __init__(self , input_size , output_size , func):
        self.norons = [noron(input_size , func) for i in range(output_size)]

    def forward(self , inputs):
        return [n.forward(inputs) for n in self.norons]

class neural_network:
    def __init__(self , input_size , hidden_layers , output_size , func):
        self.layers = []
        previous_layer_size = input_size

        for layer_size in hidden_layers:
            self.layers.append(layer(previous_layer_size , layer_size , func))
            previous_layer_size = layer_size

        self.layers.append(layer(previous_layer_size , output_size , func))

    def forward(self , inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

class neural_network_with_learning(neural_network):
    def __init__(self , input_size , hidden_layers , output_size , func , learning_rate):
        super().__init__(input_size , hidden_layers , output_size , func)
        if func == 'linear':
            self.inc_func = function_inc.linear
        elif func == 'tanh':
            self.inc_func = function_inc.tanh
        elif func == 'sigmoid':
            self.inc_func = function_inc.sigmoid
        self.learning_rate = learning_rate

    def set_weights(self , weights_bias):
        if len(weights_bias) != len(self.layers):
            raise ValueError(f'Expected {len(self.layers)} layers of weights, but got {len(weights_bias)}.')
        for layer, layer_weights in zip(self.layers, weights_bias):
            if len(layer_weights) != len(layer.norons):
                raise ValueError(f'Expected {len(layer.norons)} neurons in the layer, but got {len(layer_weights)}.')
            for noron, noron_weights_bias in zip(layer.norons, layer_weights):
                if len(noron_weights_bias) != 2:
                    raise ValueError(f'Expected 2 weights and bios for the neuron, but got {len(noron_weights_bias)}.')
                if len(noron_weights_bias[0]) != noron.input_size:
                    raise ValueError(f'Expected {noron.input_size} weights for the neuron, but got {len(noron_weights_bias[0])}.')
                noron.weights = noron_weights_bias[0]
                noron.bias = noron_weights_bias[1]

    def get_weight_list(self):
        w_of_layer = []
        for layer in self.layers:
            I = []
            for noron in layer.norons:
                I.append((noron.weights , noron.bias))
            w_of_layer.append(I)
        return w_of_layer

    def forward_layer(self , inputs):
        forwards_netwok_layer = [inputs]
        for layer in self.layers:
            inputs = layer.forward(inputs)
            forwards_netwok_layer.append(inputs)
        return forwards_netwok_layer

    def get_error_noron(self , inputs , output):
        output_layers = self.forward_layer(inputs)
        er_norons = [[0 for noron in layer.norons] for layer in self.layers]
        for index_noron in range(len(output)):
            er_norons[-1][index_noron] = ((output[index_noron] - output_layers[-1][index_noron]) ** 2)
        for index_layer in range(len(self.layers) - 1 , 0 , -1):
            for index_noron in range(len(self.layers[index_layer].norons)):
                 noron = self.layers[index_layer].norons[index_noron]
                 sum_weights = sum(noron)
                 er_noron = er_norons[index_layer][index_noron]
                 for index_weight in range(len(noron.weights)):
                        er_norons[index_layer - 1][index_weight] += (er_noron * noron.weights[index_weight] / sum_weights)
        return er_norons
    
    def train(self , inputs , output):
        output_layers = self.forward_layer(inputs)
        w_of_layer = self.get_weight_list()
        er_output = []
        for index_noron in range(len(output_layers[-1])):
            noron = self.layers[-1].norons[index_noron]
            er = 2 * (output_layers[-1][index_noron] - output[index_noron]) * self.inc_func(output_layers[-1][index_noron])
            er_output.append([(er * noron.weights[index_weight]) for index_weight in range(len(w_of_layer[-1][index_noron][0]))])
            w_new = []
            for index_weight in range(len(output_layers[-2])):
                w_old = noron.weights[index_weight]
                w_new.append(w_old - (self.learning_rate * (er * output_layers[-2][index_weight])))
            self.layers[-1].norons[index_noron].weights = w_new
            self.layers[-1].norons[index_noron].bias -= self.learning_rate * er
        for index_layer in range(len(self.layers) - 2 , -1 , -1):
            layer = self.layers[index_layer]
            new_er_output = []
            for index_noron in range(len(layer.norons)):
                noron = layer.norons[index_noron]
                er = 0
                for noron_next_layer in er_output:
                    er += noron_next_layer[index_noron]
                er *= self.inc_func(output_layers[index_layer + 1][index_noron])
                new_er_output.append([(er * noron.weights[index_weight]) for index_weight in range(len(w_of_layer[index_layer][index_noron][0]))])
                w_new = []
                for index_weight in range(len(output_layers[index_layer])):
                    w_old = noron.weights[index_weight]
                    w_new.append(w_old - (self.learning_rate * (er * output_layers[index_layer][index_weight])))
                self.layers[index_layer].norons[index_noron].weights = w_new
                self.layers[index_layer].norons[index_noron].bias -= self.learning_rate * er
            er_output = new_er_output
