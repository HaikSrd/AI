class NeuralNetwork:
    def __init__(self, input: list[float], output: list[int], output_size: int, number_of_convolution: int,
                 number_of_hidden: int, neuron_number: list[int], epochs:int,learning_rate = float):
        self.input_size = len(input[0])
        self.number_of_convolution = number_of_convolution
        self.number_of_hidden = number_of_hidden
        self.output_size = output_size
        self.hidden_input_size = self.input_size #change this after adding convolution layers
        self.total_number_layers = number_of_convolution + number_of_hidden + 1
        self.input = np.array(input)
        self.output = np.zeros((1,output_size))
        self.output[0, output] = 1
        self.learning_rate = learning_rate
        self.epochs = epochs

        # creating initial biases ///// HIDDEN LAYER BIASES ARE STORED IN self.hidden_layer_biases
        self.hidden_layer_biases = []
        hidden_layer_bias_size = neuron_number
        hidden_layer_bias_size.append(output_size)
        for i in range(self.total_number_layers):
            bias = np.zeros((1, hidden_layer_bias_size[i]))
            self.hidden_layer_biases.append(bias)

        # making the initial weights
        self.conv_layer_weights = []
        self.hidden_layer_weights = []

        #setting initial hidden weights to random numbers ///// HIDDEN LAYER WEIGHTS ARE STORED IN self.hidden_layer_weights
        weight_size_hidden = neuron_number
        weight_size_hidden.insert(0,self.hidden_input_size)
        weight_size_hidden.append(output_size)
        for i in range(number_of_hidden + 1): #added one because we are also generating the output weights
            layer = np.random.randn(weight_size_hidden[i], weight_size_hidden[i+1]) * 0.01
            #layer = np.ones((weight_size_hidden[i], weight_size_hidden[i + 1]))
            self.hidden_layer_weights.append(layer)
        print(self.hidden_layer_weights)

    def activation_function(self,input):
        return np.maximum(0, input)

    def forward_pass_hidden(self,input):
        layer_outputs = []
        layer_outputs_before_activation = []
        previous_layer = input
        for i in range(self.total_number_layers):
            if i != self.total_number_layers - 1:
                layer_output = np.array(previous_layer @ self.hidden_layer_weights[i] + self.hidden_layer_biases[i])
                layer_outputs_before_activation.append(layer_output)
                layer_outputs.append(self.activation_function(layer_output))
                previous_layer = self.activation_function(layer_output)
            else:
                layer_output = np.array(previous_layer @ self.hidden_layer_weights[i] + self.hidden_layer_biases[i])
                layer_outputs_before_activation.append(layer_output)
                layer_outputs.append(layer_output)
                previous_layer = layer_output
        return layer_outputs, layer_outputs_before_activation

    def softmax(self, numbers):
        exp_numbers = np.exp(numbers - np.max(numbers))  # For numerical stability
        return exp_numbers / np.sum(exp_numbers, axis=1, keepdims=True)  # Assuming batch input

    def cost(self, input,output):
        logits = self.forward_pass_hidden(input)[-1]
        predictions = self.softmax(logits)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)  # Avoid log(0)
        return -np.sum(output * np.log(predictions)) / len(output)


    def back_propagation(self,input,output):
        a, z = self.forward_pass_hidden(input)
        w = self.hidden_layer_weights
        dw = []
        db = [self.softmax(a[-1]) - output]
        for i in range(self.total_number_layers - 2, -1, -1):
            dw.insert(0, a[i].T @ db[0])
            db.insert(0, (db[0] @ w[i+1].T)*(z[i] > 0).astype(float))
        dw.insert(0,input.reshape(-1,1) @ db[0])
        return dw,db

    def updating_weights(self,i):
        dw,db = self.back_propagation(i)
        for i in range(self.total_number_layers):
            self.hidden_layer_weights[i] -= dw[i] * self.learning_rate
            self.hidden_layer_biases[i] -= db[i] * self.learning_rate

    def convolution(self):
        #takes an input and convolves it
        return 0

    def maxpool(self):
        return 0

    def train(self):
        for epoch in range(self.epochs):
            loss = 0
            correct_predictions = 0

            for i in range(len(self.input)):
                current_input = self.input[i]
                current_output = self.output[i]
                layer_outputs, _ = self.forward_pass_hidden(current_input,current_output)

                current_loss = self.cost(current_input)
                loss += current_loss

                if np.argmax(layer_outputs[-1]) == np.argmax(current_output):
                    correct_predictions += 1

                self.updating_weights(current_input)

            # Print the loss for every epoch and the accuracy
            average_loss = loss / len(self.input)
            accuracy = (correct_predictions / len(self.input)) * 100
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")